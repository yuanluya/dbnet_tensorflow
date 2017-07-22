import numpy as np
import os
import tensorflow as tf
import pdb

class BaseNet:
    """ Class for basic net operations and structure
    """
    def __init__(self, sess):
        self.sess = sess
        self.ys = []
        self.xs = []
        self.grad_ys = None
        self.gradients_pool = {}
        self.average_grads = {}
        self.p_grads = []
        self.p_batch_sizes = 0.0
        self.current_batch_size = None 

        #placeholders required outside for gradient accumulate
        self.batch_sizes = tf.placeholder(tf.float32, name = 'subbatch_sizes')
        self.batch_num = tf.placeholder(tf.int32, name = 'subbatch_nums')

    def accumulate(self):
        #accumulate gradients
        self.gradients = tf.gradients(self.ys, self.xs, grad_ys = self.grad_ys)

        for idx, var in enumerate(self.xs):
            self.gradients_pool[var] = tf.Variable(initial_value = np.zeros(1), 
                                                   validate_shape = False, 
                                                   trainable = False, 
                                                   dtype = tf.float32)

        def first_grad():
            ops = []
            for idx, var in enumerate(self.xs):
                ops.append(tf.assign(self.gradients_pool[var], self.gradients[idx], 
                                     validate_shape = False))
            with tf.control_dependencies(ops):
                return tf.no_op()

        def normal_grad():
            ops = []
            for idx, var in enumerate(self.xs):
                ops.append(tf.assign_add(self.gradients_pool[var], self.gradients[idx]))
            with tf.control_dependencies(ops):
                return tf.no_op()
        
        #flow_control_list = [tf.contrib.framework.
        #                        convert_to_tensor_or_sparse_tensor(grad) 
        #                     for grad in self.gradients]
        #with tf.control_dependencies(flow_control_list):
        self.accumulate_grad = tf.cond(tf.equal(self.batch_num, 0), 
                                        first_grad, normal_grad)

        #calculate final gradients for this batch
        for var in self.xs:
            self.average_grads[var] = self.gradients_pool[var] 
                                           #     tf.div(self.gradients_pool[var],
                                            #           self.batch_sizes))

    def backward(self, get_cpu_array = False):
        if get_cpu_array:
            self.p_grads = self.sess.run(list(self.average_grads.values()), 
                                         feed_dict = {self.batch_sizes: 
                                                      self.p_batch_sizes})
        else:
            self.sess.run(list(self.average_grads.values()), 
                          feed_dict = {self.batch_sizes: 
                                       self.p_batch_sizes})
        self.p_batch_sizes = 0.0
        return
    
    def get_input_gradients(self):
        return self.p_grads
