import tensorflow as tf
import numpy as np
import os
import pdb
from ..base_net.net import BaseNet

class PairNet(BaseNet):
    """ Network model for Pair Net
    Attributes:
        sess:
        max_batch_size:
        train:
    """
    def __init__(self, sess, max_batch_size, train = True):
        super(PairNet, self).__init__(sess)
        #model hyperparameters
        self.max_batch_size = max_batch_size
        self.train = train
        self.epsilon = 1e-6

        #physical inputs should be numpy arrays
        self.image_ids = None #[Total ~ M X N]
        self.text_ids = None #[Total]
        self.p_labels = None #[Total]
        self.p_loss_weights = None #[Total]
        self.p_sources = None #[Total]
        self.batch_total = tf.placeholder(tf.float32, name = 'batch_size')
        self.pos_batch_total = tf.placeholder(tf.float32,
                                              name = 'pos_batch_size')
        self.neg_batch_total = tf.placeholder(tf.float32, 
                                              name = 'neg_batch_size')
        self.res_batch_total = tf.placeholder(tf.float32, 
                                              name = 'res_batch_size')

        #physcial outputs
        self.p_loss = None
        self.p_sim = None

    def build(self, im_feat = tf.placeholder(tf.float32, name = 'im_feat'),
              dy_param = tf.placeholder(tf.float32, name = 'dy_param'),
              feature_dim = 4096, image_dropout = 0.3, 
              text_dropout = 0.3, weight_decay= 1e-7):
            
        #model parameters
        self.feature_dim = feature_dim
        self.weight_decay = weight_decay
        self.image_dropout = image_dropout
        self.text_dropout = text_dropout
        if not self.train:
            self.image_dropout = 0
            self.text_dropout = 0

        #used for data loader
        #[batch_size, feature_dim (+ 1)] for im_feat and dy_param
        self.im_feat = im_feat 
        self.dy_param = dy_param 
        self.im_idx = tf.placeholder(tf.int32, name = 'im_idx')
        self.txt_idx = tf.placeholder(tf.int32, name = 'txt_idx')
        self.labels = tf.placeholder(tf.int32, name = 'labels')
        self.loss_weights = tf.placeholder(tf.float32, name = 'loss_weights')
        self.sources = tf.placeholder(tf.int32, name = 'pair_sources')

        #######################################################################
        ########################    NETWORK STARTS    #########################
        #######################################################################
        self.im_feat_chosen = tf.gather(self.im_feat, self.im_idx)
        self.dy_param_chosen = tf.gather(self.dy_param, self.txt_idx)

        self.im_feat_dropout = tf.nn.dropout(self.im_feat_chosen, 
                                             1 - self.image_dropout)
        self.dy_param_decay = self.weight_decay * tf.to_double(tf.nn.l2_loss(self.dy_param_chosen))
        
        #prepare kernel
        self.dy_kernel = tf.slice(self.dy_param_chosen, 
                                  [0, 0], [-1, self.feature_dim])
        self.dy_bias = tf.slice(self.dy_param_chosen, 
                                [0, self.feature_dim], [-1, 1])
        self.dy_kernel_dropout = tf.nn.dropout(self.dy_kernel, 1 - self.text_dropout)
        
        #get binary classification score
        self.cls_pre = tf.reduce_sum(tf.multiply(self.im_feat_dropout, 
                                                 self.dy_kernel_dropout), 1)
        self.cls_single = tf.add(tf.expand_dims(self.cls_pre, -1), self.dy_bias)
        self.cls = tf.concat([-1 * self.cls_single, self.cls_single], axis = 1) 
        #logit = lambda x: np.log(x) - np.log(1 - x)
        #self.cls = tf.clip_by_value(tf.add(tf.expand_dims(self.cls_pre, -1), self.dy_bias), logit(self.epsilon), logit(1 - self.epsilon))
        
        self.sim = tf.slice(tf.nn.softmax(self.cls), [0, 1], [-1, 1])
        self.full_labels = tf.one_hot(self.labels, 2) 
        losses = tf.nn.softmax_cross_entropy_with_logits(
            labels = self.full_labels, logits = self.cls)
        self.losses = tf.expand_dims(tf.multiply(self.loss_weights, losses), -1)
        self.pos_mask = tf.expand_dims(tf.equal(self.sources, 1), -1)
        self.neg_mask = tf.expand_dims(tf.equal(self.sources, 0), -1)
        self.rest_mask = tf.expand_dims(tf.equal(self.sources, 2), -1)

        pos_mask = tf.cast(self.pos_mask, tf.float32)
        neg_mask = tf.cast(self.neg_mask, tf.float32)
        rest_mask = tf.cast(self.rest_mask, tf.float32)
        
        self.pos_loss = tf.reduce_sum(tf.multiply(self.losses, pos_mask)) / self.pos_batch_total 
        self.neg_loss = tf.reduce_sum(tf.multiply(self.losses, neg_mask)) / self.neg_batch_total
        self.rest_loss = tf.reduce_sum(tf.multiply(self.losses, rest_mask)) / self.res_batch_total
        self.cls_loss = self.pos_loss + self.neg_loss + self.rest_loss

        #accumulate gradients
        self.current_batch_size = tf.to_float(tf.shape(self.im_feat_chosen)[0])
        self.loss = tf.to_float(self.dy_param_decay) + self.cls_loss 
        self.xs = [self.im_feat, self.dy_param]
        self.ys = [self.loss]
        self.accumulate()

    def set_input(self, image_ids, text_ids, labels = None, 
                  loss_weights = None, sources = None):
        self.image_ids = image_ids
        self.text_ids = text_ids
        self.p_labels = labels
        self.p_loss_weights = loss_weights
        self.p_sources = sources

    def get_output(self):
        return (self.p_loss, self.p_sim, self.p_pos_loss, self.p_neg_loss, 
                self.p_rest_loss)

    def forward(self, compute_grads = True, compute_loss = True):
        #initialize accumulating variables for this batch
        current_im_idx = 0
        self.p_loss = 0
        self.p_pos_loss = 0
        self.p_neg_loss = 0
        self.p_rest_loss = 0
        self.p_decay = 0
        self.p_sim = np.zeros((0, 1))

        #set parameters for this batch
        total_pos = np.sum(self.p_sources == 1)
        assert total_pos == np.sum(self.p_labels == 1)
        total_neg = np.sum(self.p_sources == 0)
        total_res = np.sum(self.p_sources == 2)
        self.p_batch_sizes = self.text_ids.shape[0] * 1.0
        
        self.batch_size = self.max_batch_size
        #start accumulate gradients for subbatches
        while current_im_idx < self.text_ids.shape[0]:
            if compute_grads:
                [p_loss, p_pos_loss, p_neg_loss, p_rest_loss, dy_param_decay, p_sim, _] = (
                    self.sess.run(
                        [self.loss, self.pos_loss, self.neg_loss, self.rest_loss, self.dy_param_decay, self.sim, self.accumulate_grad],
                        feed_dict = {
                            self.im_idx: 
                                self.image_ids[current_im_idx: current_im_idx + self.batch_size],
                            self.txt_idx: 
                                self.text_ids[current_im_idx: current_im_idx + self.batch_size],
                            self.labels: 
                                self.p_labels[current_im_idx: current_im_idx + self.batch_size],
                            self.loss_weights: 
                                self.p_loss_weights[current_im_idx: current_im_idx + self.batch_size],
                            self.sources: 
                                self.p_sources[current_im_idx: current_im_idx + self.batch_size],
                            self.batch_total: self.p_batch_sizes,
                            self.pos_batch_total: total_pos, 
                            self.neg_batch_total: total_neg,
                            self.res_batch_total: total_res,
                            self.batch_num: current_im_idx}))
            elif compute_loss:
                [p_loss, p_pos_loss, p_neg_loss, p_rest_loss, dy_param_decay, p_sim] = (
                    self.sess.run(
                        [self.loss, self.pos_loss, self.neg_loss, 
                         self.rest_loss, self.dy_param_decay, self.sim], 
                        feed_dict = {
                            self.im_idx: 
                                self.image_ids[current_im_idx: 
                                               current_im_idx + 
                                               self.batch_size],
                            self.txt_idx: 
                                self.text_ids[current_im_idx: 
                                              current_im_idx + 
                                              self.batch_size],
                            self.labels: 
                                self.p_labels[current_im_idx: 
                                              current_im_idx + 
                                              self.batch_size],
                            self.loss_weights: 
                                self.p_loss_weights[current_im_idx: 
                                                    current_im_idx + 
                                                    self.batch_size],
                            self.sources: 
                                self.p_sources[current_im_idx: 
                                               current_im_idx + 
                                               self.batch_size],
                            self.batch_total: self.p_batch_sizes, 
                            self.pos_batch_total: total_pos, 
                            self.res_batch_total: total_res,
                            self.neg_batch_total: total_neg}))
            else:
                [p_sim] = self.sess.run([self.sim],
                        feed_dict = {
                            self.im_idx: 
                                self.image_ids[current_im_idx: 
                                               current_im_idx + 
                                               self.batch_size],
                            self.txt_idx: 
                                self.text_ids[current_im_idx: 
                                              current_im_idx + 
                                              self.batch_size]})
                p_loss = None
            if compute_loss:
                self.p_loss += p_loss
                self.p_pos_loss += p_pos_loss 
                self.p_neg_loss += p_neg_loss
                self.p_rest_loss += p_rest_loss
                self.p_decay += dy_param_decay
            self.p_sim = np.concatenate((self.p_sim, p_sim))
            current_im_idx += self.batch_size
            
            #avoid small subbatch
            if self.p_batch_sizes > current_im_idx + self.batch_size and\
               self.p_batch_sizes - current_im_idx - self.batch_size < 0.25 * self.batch_size:
                self.batch_size = int(self.p_batch_sizes - current_im_idx)
        return
