import numpy as np
import inspect
import os
import tensorflow as tf
from ..roi_pooling_layer.roi_pooling_op import roi_pool
from ..roi_pooling_layer.roi_pooling_op_grad import * 

class Vgg16:
    def __init__(self, lr, RegionNet_npy_path = 'frcnn_Region_Feat_Net.npy', train = True):
        #load saved model
        try:
            path = inspect.getfile(Vgg16)
            path = os.path.abspath(os.path.join(path, os.pardir))
            RegionNet_npy_path = os.path.join(path, RegionNet_npy_path)
            self.data_dict = np.load(RegionNet_npy_path, encoding='latin1').item()
            print("Image Feat Net npy file loaded")
        except:
            print('[WARNING!]Image Feat Net npy file not found,'
                'we don\'t recommend training this network from scratch')
            self.data_dict = {}
        self.lr = lr
        self.train = train
        self.varlist_conv = []
        self.varlist_region = []
        self.net_type = 'Vgg16'

    def build(self, bgr, rois, parameters):
        #set placeholder
        self.bgr = bgr
        self.rois = rois

        #set parameters
        self.feature_dim = parameters['feature_dim']
        self.weight_decay = parameters['weight_decay']
        self.dropout_ratio = parameters['dropout_ratio']
        self.dropout_flag = parameters['dropout_flag']
        self.roi_size = parameters['roi_size']
        self.roi_scale = parameters['roi_scale']
        self.build_conv()
        self.build_region()
        return self.relu7

    def build_conv(self):
        """
        load variable from npy to build the VGG

        :param bgr: bgr image [batch, height, width, 3] values scaled [0, 1]
        """
        # Convert RGB to BGR
        self.conv1_1 = self.conv_layer(self.bgr, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self.max_pool(self.conv4_3, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")

    def build_region(self):
        [self.rois_feat, _] = roi_pool(self.conv5_3, self.rois,
                                       self.roi_size, self.roi_size, 
                                       self.roi_scale)
        
        # reshape tensor so that every channel's map are expanded 
        # with rows unchanged
        conv_channels = self.rois_feat.get_shape().as_list()[-1]
        self.rois_feat_reshape1 = tf.reshape(self.rois_feat, 
                                             [-1, self.roi_size ** 2,
                                              conv_channels])
        self.rois_feat_transpose = tf.transpose(self.rois_feat_reshape1, 
                                                perm = [0, 2, 1])
        self.rois_feat_reshape2 = tf.reshape(self.rois_feat_transpose, 
                                             [-1, self.roi_size ** 2 * 
                                                  conv_channels])

        self.fc6 = self.fc_layer(self.rois_feat_reshape2, 'fc6', 
                                 [self.roi_size ** 2 * 512, 4096])
        self.relu6 = tf.nn.relu(self.fc6)

        #hand write dropout
        if self.train:
            self.relu6 = dropout(self.relu6, self.dropout_flag, 
                                 self.dropout_ratio, 'fc6_dropout') 

        self.fc7 = self.fc_layer(self.relu6, "fc7", [4096, self.feature_dim])
        self.relu7 = tf.nn.relu(self.fc7)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias_conv(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def get_conv_filter(self, name):
        var = tf.Variable(self.data_dict[name]['weights'], name="filter",
                          trainable = (self.lr > 0), dtype = tf.float32)
        wd = tf.multiply(tf.nn.l2_loss(var), self.weight_decay, name = 'weight_decay') 
        tf.add_to_collection('img_net_weight_decay', wd)
        self.varlist_conv.append(var)
        return var

    def get_bias_conv(self, name):
        var = tf.Variable(self.data_dict[name]['biases'], name="biases",
                          trainable = (self.lr > 0), dtype = tf.float32)
        self.varlist_conv.append(var)
        return var

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def fc_layer(self, bottom, name, shape):
        with tf.variable_scope(name) as scope:
            weights = self.get_fc_weight(name, shape)
            biases = self.get_bias(name, [shape[1]])

        # Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.
        fc = tf.nn.bias_add(tf.matmul(bottom, weights), biases)

        return fc

    def get_bias(self, name, shape):
        if self.data_dict.get(name):
            init = tf.constant_initializer(
                value = self.data_dict[name]['biases'], dtype = tf.float32)
        else:
            init = tf.constant_initializer(0.015)
            print('[WARNING]Region Feat Net %s layer\'s bias are random init '
                  'with shape [%d]' % (name, shape[0]))

        var = tf.get_variable(name = 'bias', initializer = init, 
                              shape = shape, dtype = tf.float32)
        self.varlist_region.append(var)
        return var

    def get_fc_weight(self, name, shape):
        if self.data_dict.get(name):
            init = tf.constant_initializer(
                value = self.data_dict[name]['weights'], dtype = tf.float32)
        else:
            init = tf.random_normal_initializer(mean = 0.0, stddev = 0.0005)
            print('[WARNING]Region Feat Net %s layer\'s weights are '
                  'random init!' % name)

        var = tf.get_variable(name = 'weights', initializer = init, 
                              shape = shape, dtype = tf.float32)
        weight_decay = tf.multiply(tf.nn.l2_loss(var), self.weight_decay, 
                                   name = 'weight_decay')
        tf.add_to_collection('img_net_weight_decay', weight_decay)
        self.varlist_region.append(var)
        return var

def dropout(bottom, random_flag, dropout_ratio = 0.5, name = 'dropout'):
    with tf.variable_scope(name):
        drop_mask_r = tf.random_uniform(shape = tf.shape(bottom))
        drop_mask_r = tf.cast(tf.greater(drop_mask_r, dropout_ratio), 
                              tf.float32)
        drop_mask_v = tf.Variable(initial_value = np.zeros(1), 
                validate_shape = False, trainable = False, dtype = tf.float32)
        assign_dropout = tf.assign(drop_mask_v, drop_mask_r, 
                                   validate_shape = False)
        assign_dropout = tf.cond(tf.equal(random_flag, 1),
                lambda: tf.assign(drop_mask_v, drop_mask_r, 
                                  validate_shape = False),
                lambda: tf.identity(drop_mask_v))
        return tf.div(tf.multiply(bottom, assign_dropout), (1 - dropout_ratio))

