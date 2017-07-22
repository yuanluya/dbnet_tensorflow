import tensorflow as tf
import numpy as np
import pdb
from .image_feat_net.vgg16.net import Vgg16
from .image_feat_net.resnet101.net import Resnet101
from .image_feat_net.net import ImageFeatNet 
from .text_feat_net.net import TextFeatNet 
from .pair_net.net import PairNet 
from .base import Model

class NetWrapper(Model):
    """
    The class is for wrapping TextFeatNet, ImageFeatNet and PairNet
    together and provide a uniform interface for the network training
    steps.
    Attributes:
        sess: Tensorflow session
        opt_image: optimizer for image end
        opt_text: optimizer for text end
        train: a boolean indicating if the network is currently in the 
            training phase, default is True
        pair_net_max_batch_size: an integer indicating the maximum batch
            size of the pair net, default is 500
    """
    def __init__(self, sess, image_net_type, image_lr_conv, image_lr_region, text_lr,
                 pair_net_max_batch_size, train, image_init_npy, text_init_npy):
                 
        self.image_net_type = image_net_type
        self.image_lr_conv = image_lr_conv
        self.image_lr_region = image_lr_region
        self.text_lr = text_lr

        self.sess = sess
        self.train = (train == 'train')
        self.pair_net = PairNet(sess, pair_net_max_batch_size, 
                                train = self.train)
        self.image_net = ImageFeatNet(sess, self.image_lr_conv,
                                      self.image_lr_region, train = self.train)
        self.text_net_opt_type = 'Adam'
        if self.image_lr_conv != 0 or self.image_lr_region != 0:
            self.text_net_opt_type = 'SGD'
        self.text_net = TextFeatNet(sess, self.text_lr, train = self.train,
            opt_type = self.text_net_opt_type, TextNet_npy_path = text_init_npy)
        if self.image_net_type == 'resnet101':
            self.text_feature_dim = 2049
            self.im_sub_net = Resnet101(RegionNet_npy_path = image_init_npy, train = self.train)
        elif self.image_net_type == 'vgg16': 
            self.text_feature_dim = 4097
            self.im_sub_net = Vgg16(self.image_lr_conv,
                RegionNet_npy_path = image_init_npy, train = self.train)
        self.data_dict = None
        self.varlist = None

    def set_input(self, data):
        self.data_dict = data

    def build(self):
        with tf.variable_scope('Text_Network'):
            self.text_net.build(output_feature_dim = self.text_feature_dim)

        with tf.variable_scope('Image_Network'):
            if self.image_net_type == 'resnet101':
                self.image_net.build(self.im_sub_net, feature_dim = 2048, roi_size = 14)
            else:
                self.image_net.build(self.im_sub_net)

        with tf.variable_scope('Pair_Network'):
            self.pair_net.build(im_feat = self.image_net.output, 
                                dy_param = self.text_net.output,
                                feature_dim = self.text_feature_dim - 1)

        self.image_net.output_grad = (
            self.pair_net.gradients_pool[self.image_net.output])
        self.text_net.output_grad = (
            self.pair_net.gradients_pool[self.text_net.output])

        self.image_net.accumulate()
        self.text_net.accumulate()
        self.varlist = self.image_net.sub_net.varlist_conv\
                     + self.image_net.sub_net.varlist_region\
                     + self.text_net.varlist\
                     + self.text_net.varlist_relu

    def forward(self, compute_grads = True, compute_loss = True):
        self.image_net.set_input(self.data_dict['images'], 
                                 self.data_dict['rois'])
        self.image_net.forward()
        self.text_net.set_input(self.data_dict['phrases'])
        self.text_net.forward()
        self.pair_net.set_input(self.data_dict['roi_ids'], 
                                self.data_dict['phrase_ids'],
                                self.data_dict['labels'], 
                                self.data_dict['loss_weights'], 
                                self.data_dict['sources'])
        self.pair_net.forward(compute_grads, compute_loss)

    def backward(self):
        self.pair_net.backward()
        self.text_net.backward()
        self.image_net.backward()
        
    def forward_backward(self):
        self.forward()
        self.backward()
    
    def get_output(self, current_iter = 0):
        self.output = self.pair_net.get_output()
        if current_iter is not 0:
            self.show_result(current_iter)
        return self.output
    
    def show_result(self, current_iter):
        self.prediction = self.output[1] > 0.5
        total_pos = np.sum(self.data_dict['labels'] == 1)
        total_predict = np.sum(self.prediction == 1)
        self.recall = (np.sum((self.data_dict['labels'] == 1) * 
                              (self.data_dict['labels'] == 
                               self.prediction[:, 0])) 
                       / total_pos)
        self.precision = (np.sum((self.data_dict['labels'] == 1) * 
                                 (self.data_dict['labels'] == 
                                  self.prediction[:, 0]))
                          / total_predict)
        #print results
        print('Iter: %d' % current_iter)
        print('Looked images:', self.data_dict['image_ids'])
        print('\t[$$]Precision: %f, Recall: %f' % (self.precision, 
                                                   self.recall))
        print('\t[TL] Total loss is       %f' % self.output[0])
        print('\t[PL]Raw positive loss is %f' % self.output[2])
        print('\t[NL]Raw negative loss is %f' % self.output[3])
        print('\t[RL]Raw rest loss is     %f\n' % self.output[4])
