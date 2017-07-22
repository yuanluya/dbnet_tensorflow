import numpy as np
import inspect
import os 
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages
from ..roi_pooling_layer.roi_pooling_op import roi_pool
from ..roi_pooling_layer.roi_pooling_op_grad import * 
import pdb

MOVING_AVERAGE_DECAY = 0.9997
BN_DECAY = MOVING_AVERAGE_DECAY
BN_EPSILON = 0.001
CONV_WEIGHT_DECAY = 0.00004
CONV_WEIGHT_STDDEV = 0.1
FC_WEIGHT_DECAY = 0.00004
FC_WEIGHT_STDDEV = 0.01
UPDATE_OPS_COLLECTION = 'resnet_update_ops'  # must be grouped with training op

class Resnet101:
    def __init__(self, RegionNet_npy_path='frcnn_Region_Feat_Net.npy', train=True):
        #load saved model
        try:
            path = inspect.getfile(Resnet101)
            path = os.path.abspath(os.path.join(path, os.pardir))
            RegionNet_npy_path = os.path.join(path, RegionNet_npy_path)
            self.data_dict = np.load(RegionNet_npy_path, encoding='latin1').item()
            print("Image Feat Net npy file loaded")
        except:
            print('[WARNING!]Image Feat Net npy file not found,'
                'we don\'t recommend training this network from scratch')
            self.data_dict = {}
        self.is_training = train 
        self.varlist_conv = []
        self.varlist_region = []
        self.net_type = 'Resnet101'
        self.activation = tf.nn.relu
    
    def build(self, bgr, rois, parameters,
              num_blocks=[3, 4, 23, 3],
              use_bias=False):

        self.bgr = bgr
        self.rois = rois
        self.weight_decay = parameters['weight_decay']
        self.roi_size = parameters['roi_size']
        self.roi_scale = parameters['roi_scale']

        c = {}
        c['bottleneck'] = True
        c['is_training'] = tf.convert_to_tensor(self.is_training,
                                                dtype='bool',
                                                name='is_training')
        c['ksize'] = 3
        c['stride'] = 1
        c['use_bias'] = use_bias
        c['num_blocks'] = num_blocks
        c['stack_stride'] = 2

        with tf.variable_scope('scale1'):
            c['conv_filters_out'] = 64
            c['ksize'] = 7
            c['stride'] = 2
            x = self.conv(self.bgr, c, 'conv1')
            x = self.bn(x, c, 'bn_conv1')
            self.scale1_feat = self.activation(x)

        with tf.variable_scope('scale2'):
            x = self._max_pool(self.scale1_feat, ksize=3, stride=2)
            c['num_blocks'] = num_blocks[0]
            c['stack_stride'] = 1
            c['block_filters_internal'] = 64
            self.scale2_feat = self.stack(x, c, '2')

        with tf.variable_scope('scale3'):
            c['num_blocks'] = num_blocks[1]
            c['block_filters_internal'] = 128
            c['stack_stride'] = 2
            self.scale3_feat = self.stack(self.scale2_feat, c, '3')

        with tf.variable_scope('scale4'):
            c['num_blocks'] = num_blocks[2]
            c['block_filters_internal'] = 256
            assert c['stack_stride'] == 2
            self.scale4_feat = self.stack(self.scale3_feat, c, '4')

        
        [self.rois_feat, _] = roi_pool(self.scale4_feat, self.rois,
                                       self.roi_size, self.roi_size, 
                                       self.roi_scale)

        with tf.variable_scope('scale5'):
            c['num_blocks'] = num_blocks[3]
            c['block_filters_internal'] = 512
            assert c['stack_stride'] == 2
            self.scale5_feat = self.stack(self.rois_feat, c, '5', belong='region')

        # post-net
        self.final_feature = tf.reduce_mean(self.scale5_feat, reduction_indices=[1, 2], name="avg_pool")

        return self.final_feature

    def stack(self, x, c, stack_caffe_scale, belong='conv'):
        if c['num_blocks'] == 3:
            block_names = ['a', 'b', 'c']
        else:
            block_names = ['a'] + ['b' + str(i + 1) for i in range(c['num_blocks'] - 1)]
        for n in range(c['num_blocks']):
            s = c['stack_stride'] if n == 0 else 1
            c['block_stride'] = s
            with tf.variable_scope('block%d' % (n + 1)):
                x = self.block(x, c, stack_caffe_scale+block_names[n], belong)
        return x


    def block(self, x, c, block_caffe_name, belong='conv'):
        filters_in = x.get_shape()[-1]

        # Note: filters_out isn't how many filters are outputed. 
        # That is the case when bottleneck=False but when bottleneck is 
        # True, filters_internal*4 filters are outputted. filters_internal is how many filters
        # the 3x3 convs output internally.
        m = 4 if c['bottleneck'] else 1
        filters_out = m * c['block_filters_internal']

        shortcut = x  # branch 1

        c['conv_filters_out'] = c['block_filters_internal']

        with tf.variable_scope('a'):
            c['ksize'] = 1
            c['stride'] = c['block_stride']
            x = self.conv(x, c, 'res'+block_caffe_name+'_branch2a', belong)
            x = self.bn(x, c, 'bn'+block_caffe_name+'_branch2a', belong)
            x = self.activation(x)

        with tf.variable_scope('b'):
            c['ksize'] = 3
            c['stride'] = 1 
            x = self.conv(x, c, 'res'+block_caffe_name+'_branch2b', belong)
            x = self.bn(x, c, 'bn'+block_caffe_name+'_branch2b', belong)
            x = self.activation(x)

        with tf.variable_scope('c'):
            c['conv_filters_out'] = filters_out
            c['ksize'] = 1
            assert c['stride'] == 1
            x = self.conv(x, c, 'res'+block_caffe_name+'_branch2c', belong)
            x = self.bn(x, c, 'bn'+block_caffe_name+'_branch2c', belong)

        with tf.variable_scope('shortcut'):
            if filters_out != filters_in or c['block_stride'] != 1:
                c['ksize'] = 1
                c['stride'] = c['block_stride']
                c['conv_filters_out'] = filters_out
                shortcut = self.conv(shortcut, c, 'res'+block_caffe_name+'_branch1', belong)
                shortcut = self.bn(shortcut, c, 'bn'+block_caffe_name+'_branch1', belong)

        return self.activation(x + shortcut)


    def bn(self, x, c, caffe_name, belong='conv'):
        x_shape = x.get_shape()
        params_shape = x_shape[-1:]

        if c['use_bias']:
            bias = self._get_variable('bias', params_shape,
                                initializer=tf.zeros_initializer())
            return x + bias


        axis = list(range(len(x_shape) - 1))

        beta = self._get_variable('beta',
                            caffe_name,
                            params_shape,
                            key='offset',
                            initializer=tf.zeros_initializer())
        gamma = self._get_variable('gamma',
                            caffe_name,
                            params_shape,
                            key='scale',
                            initializer=tf.ones_initializer())
        if belong == 'conv':
            self.varlist_conv.extend([beta, gamma])
        elif belong == 'region':
            self.varlist_region.extend([beta, gamma])

        moving_mean = self._get_variable('moving_mean',
                                    caffe_name,
                                    params_shape,
                                    key='mean',
                                    initializer=tf.zeros_initializer(),
                                    trainable=False)
        moving_variance = self._get_variable('moving_variance',
                                        caffe_name,
                                        params_shape,
                                        key='variance',
                                        initializer=tf.ones_initializer(),
                                        trainable=False)

        # These ops will only be preformed when training.
        mean, variance = tf.nn.moments(x, axis)
        update_moving_mean = moving_averages.assign_moving_average(moving_mean,
                                                                mean, BN_DECAY)
        update_moving_variance = moving_averages.assign_moving_average(
            moving_variance, variance, BN_DECAY)
        tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
        tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)

        mean, variance = control_flow_ops.cond(
            c['is_training'], lambda: (mean, variance),
            lambda: (moving_mean, moving_variance))

        x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)
        #x.set_shape(inputs.get_shape()) ??

        return x

    def _get_variable(self,
                    name,
                    caffe_name,
                    shape,
                    initializer,
                    key='weights',
                    dtype='float',
                    trainable=True):

        "A little wrapper around tf.get_variable to do weight decay and add to"
        "resnet collection"
        if self.data_dict.get(caffe_name):
            initializer = tf.constant_initializer(value = self.data_dict[caffe_name][key], dtype = tf.float32)
        else:
            print('[WARNING] Resnet block with caffe name\
                %s:%s was initialized by random' % (caffe_name, key))
        var = tf.get_variable(name, shape=shape, initializer=initializer,
                              dtype=dtype, trainable=trainable)
        if self.weight_decay > 0:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), self.weight_decay, 
                                    name = 'weight_decay')
            tf.add_to_collection('img_net_weight_decay', weight_decay)
        return var 

    def conv(self, x, c, caffe_name, belong='conv'):
        ksize = c['ksize']
        stride = c['stride']
        filters_out = c['conv_filters_out']

        filters_in = x.get_shape()[-1]
        shape = [ksize, ksize, filters_in, filters_out]
        initializer = tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV)
        weights = self._get_variable('weights',
                                caffe_name,
                                shape=shape,
                                dtype='float32',
                                initializer=initializer)
        if belong == 'conv':
            self.varlist_conv.append(weights)
        elif belong == 'region':
            self.varlist_region.append(weights)
        return tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')


    def _max_pool(self, x, ksize=3, stride=2):
        return tf.nn.max_pool(x,
                            ksize=[1, ksize, ksize, 1],
                            strides=[1, stride, stride, 1],
                            padding='SAME')
