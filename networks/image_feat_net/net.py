import tensorflow as tf
import numpy as np
import os 
import inspect

class ImageFeatNet:
    """ Network model for Image Feat Net
    Attributes:
        sess:
        opt:
        max_batch_size:
        train:
        RegionNet_npy_path:
    """
    def __init__(self, sess, lr_conv, lr_region, max_batch_size = 32, train = True):
        self.lr_conv = lr_conv
        self.lr_region = lr_region
        self.opt_conv = tf.train.MomentumOptimizer(self.lr_conv, 0.9)
        self.opt_region = tf.train.MomentumOptimizer(self.lr_region, 0.9)
        #model hyperparameters
        self.sess = sess
        self.max_batch_size = max_batch_size
        self.train = train

        #physical inputs should be numpy arrays
        self.images = tf.placeholder(tf.float32, shape = [None, None, None, 3],
                                     name = 'image_inputs')
        #[batch_idx, xmin, ymin, xmax, ymax]
        self.rois = tf.placeholder(tf.float32, shape = [None, 5], 
                                   name = 'roi_inputs')
        self.dropout_flag = tf.placeholder(tf.int32)
        self.p_images = None
        self.p_rois = None

        #physcial outputs
        self.p_region_feats = None

    def build(self, sub_net, output_grad = tf.placeholder(tf.float32),
              feature_dim = 4096, roi_size = 7, roi_scale = 0.0625,
              dropout_ratio = 0.3, weight_decay= 1e-4, batch_size = 16):
    
        #conv net base
        self.sub_net = sub_net
        self.output_grad = output_grad

        #optimization utility
        self.batch_size = batch_size
        assert self.batch_size < self.max_batch_size

        #model parameters
        self.parameters = {}
        self.parameters['feature_dim'] = feature_dim
        self.parameters['weight_decay'] = weight_decay
        self.parameters['dropout_ratio'] = dropout_ratio
        self.parameters['roi_size'] = roi_size
        self.parameters['roi_scale'] = roi_scale
        self.parameters['dropout_flag'] = self.dropout_flag

        #######################################################################
        ########################    NETWORK STARTS    #########################
        #######################################################################
        self.roi_features = self.sub_net.build(self.images, self.rois, self.parameters)
        self.output = tf.Variable(initial_value = 1.0, trainable = False, 
                                  validate_shape = False, dtype = tf.float32)
        self.get_output = tf.assign(self.output, self.roi_features, 
                                    validate_shape = False)

        #gather weight decays
        self.wd = tf.add_n(tf.get_collection('img_net_weight_decay'), 
                           name = 'img_net_total_weight_decay')
        if self.sub_net.net_type == 'Vgg16':
            self.extra_update = [tf.no_op()]
        elif self.sub_net.net_type == 'Resnet101':
            self.extra_update = tf.get_collection('resnet_update_ops')


    def accumulate(self):
        self.ys = [self.wd, self.roi_features]
        self.grad_ys = [1.0, self.output_grad]

        self.gradients_conv = tf.gradients(self.ys, self.sub_net.varlist_conv, grad_ys = self.grad_ys)
        self.gradients_region = tf.gradients(self.ys, self.sub_net.varlist_region, grad_ys = self.grad_ys)
        
        self.grad_and_vars_conv = []
        self.grad_and_vars_region = []
        
        for idx, var in enumerate(self.sub_net.varlist_conv):
            self.grad_and_vars_conv.append((self.gradients_conv[idx], var))
        for idx, var in enumerate(self.sub_net.varlist_region):
            self.grad_and_vars_region.append((self.gradients_region[idx], var))

       	#apply gradients 
        with tf.control_dependencies(self.gradients_conv + self.gradients_region):
            self.train_op = tf.group(self.opt_conv.apply_gradients(self.grad_and_vars_conv),
									 self.opt_region.apply_gradients(self.grad_and_vars_region), *self.extra_update)

    def set_input(self, images, rois):
        self.p_images = images
        self.p_rois = rois

    def get_output(self):
        return self.p_roi_features

    def forward(self, physical_output = False):
        if physical_output:
            [self.p_roi_features] = self.sess.run([self.get_output], 
                                           feed_dict = {
                                               self.images: self.p_images, 
                                               self.rois: self.p_rois, 
                                               self.dropout_flag: 1})
        else:
            self.sess.run([self.get_output],
                          feed_dict = {
                              self.images: self.p_images, 
                              self.rois: self.p_rois, 
                              self.dropout_flag: 1})
        return
    
    def backward(self):
        self.sess.run([self.train_op], 
                      feed_dict = {
                          self.images: self.p_images, 
                          self.rois: self.p_rois, 
                          self.dropout_flag: 0})
        return  

