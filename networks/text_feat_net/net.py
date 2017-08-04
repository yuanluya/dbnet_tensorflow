import tensorflow as tf
import numpy as np
import os
import pdb
import inspect

class TextFeatNet:
    """ Network for Text Feat Net
    Attributes:
        sess: Tensorflow session
        opt: Optimizer
        max_batch_size:
        train:
        sequence_length:
        TextNet_npy_path:
    """
    def __init__(self, sess, lr, max_batch_size = 32, 
                 train = True, sequence_length = 256, opt_type = 'Adam', 
                 TextNet_npy_path = 'Text_Feat_Net.npy'):
        #load saved model
        try:
            path = inspect.getfile(TextFeatNet)
            path = os.path.abspath(os.path.join(path, os.pardir))
            TextNet_npy_path = os.path.join(path, TextNet_npy_path)
            self.data_dict = np.load(TextNet_npy_path, encoding='latin1').item()
            print("Text Feat Net npy file loaded")
        except:
            self.data_dict = {}
            print('[WARNING]Text Feat load file not found, train from scratch!')
        self.lr = lr
        self.backward_counter = 0
        self.opt_type = opt_type
        if self.opt_type == 'SGD': 
            self.opt = tf.train.GradientDescentOptimizer(self.lr)
            self.opt_relu = tf.train.GradientDescentOptimizer(0.1 * self.lr)
        elif self.opt_type == 'Adam':
            self.opt = tf.train.AdamOptimizer(self.lr)
            self.opt_relu = tf.train.AdamOptimizer(0.1 * self.lr)
        self.varlist = []
        self.varlist_relu = []
        #model hyperparameters
        self.sess = sess
        self.sequence_length = sequence_length
        self.max_batch_size = max_batch_size
        self.train = train

        #physcial outputs
        self.p_dy_param = None

    def build(self, output_grad = tf.placeholder(tf.float32), 
              input_feature_dim = 74, weight_decay= 5e-4, 
              relu_coef = 0.1, batch_size = 16, 
              output_feature_dim = 4097):
    
        #optimization utility
        self.output_grad = output_grad

        self.batch_size = batch_size
        assert self.batch_size < self.max_batch_size

        #model parameters
        self.input_feature_dim = input_feature_dim
        self.output_feature_dim = output_feature_dim
        self.weight_decay = weight_decay
        self.relu_coef = relu_coef 

        #physical inputs should be numpy arrays
        self.texts = tf.placeholder(tf.float32, 
                                    shape = [None, 1, self.sequence_length,
                                             self.input_feature_dim])
        self.p_texts = None

        #######################################################################
        ########################    NETWORK STARTS    #########################
        #######################################################################

        #conv1
        self.conv1 = self.conv_layer(self.texts, 'conv1', 
                                     [1, 7, self.input_feature_dim, 256])
        self.conv1_relu = self.lrelu(self.conv1, self.relu_coef, 'conv1_relu')
        self.conv1_pool = tf.nn.max_pool(self.conv1_relu, 
                                         [1, 1, 2, 1], [1, 1, 2, 1], 
                                         padding = 'VALID') 
        
        #conv2
        self.conv2_1 = self.conv_layer(self.conv1_pool, 'conv2_1', 
                                       [1, 7, 256, 256])
        self.conv2_1_relu = self.lrelu(self.conv2_1, self.relu_coef, 'conv2_1_relu')
        self.conv2_2 = self.conv_layer(self.conv2_1_relu, 'conv2_2', 
                                       [1, 3, 256, 256])
        self.conv2_2_relu = self.lrelu(self.conv2_2, self.relu_coef, 'conv2_2_relu') 
        self.conv2_3 = self.conv_layer(self.conv2_2_relu, 'conv2_3', 
                                       [1, 3, 256, 256])
        self.conv2_3_relu = self.lrelu(self.conv2_3, self.relu_coef, 'conv2_3_relu') 
        self.conv2_3_pool = tf.nn.max_pool(self.conv2_3_relu, 
                                           [1, 1, 2, 1], [1, 1, 2, 1], 
                                           padding = 'VALID', 
                                           name = 'conv2_3_pooling') 

        #conv3
        self.conv3_1 = self.conv_layer(self.conv2_3_pool, 'conv3_1', 
                                       [1, 3, 256, 512])
        self.conv3_1_relu = self.lrelu(self.conv3_1, self.relu_coef, 'conv3_1_relu') 
        self.conv3_2 = self.conv_layer(self.conv3_1_relu, 'conv3_2', 
                                       [1, 3, 512, 512])
        self.conv3_2_relu = self.lrelu(self.conv3_2, self.relu_coef, 'conv3_2_relu') 
        self.conv3_2_pool = tf.nn.max_pool(self.conv3_2_relu, 
                                           [1, 1, 2, 1], [1, 1, 2, 1], 
                                           padding = 'VALID', 
                                           name = 'conv3_2_pooling') 

        #fully connected
        expand_size = self.conv3_2_pool.get_shape().as_list()
        self.conv3_2_reshape_1 = tf.reshape(self.conv3_2_pool,
                [-1, expand_size[1] * expand_size[2], expand_size[3] ])
        self.conv3_2_transpose = tf.transpose(self.conv3_2_reshape_1, [0, 2, 1])
        self.conv3_2_reshape_2 = tf.reshape(self.conv3_2_transpose,
                [-1, expand_size[1] * expand_size[2] * expand_size[3]])

        self.fc4 = self.fc_layer(self.conv3_2_reshape_2, 'fc4', 2048, 0.1, 0.01)
        self.fc4_relu = self.lrelu(self.fc4, self.relu_coef, 'fc4_relu')
        
        #dynamic filters
        self.pre_dy_fc1 = self.fc_layer(self.fc4_relu, 'pre_dy_fc1', 2048, bias_decay = True)
        self.pre_dy_fc1_relu = self.lrelu(self.pre_dy_fc1, self.relu_coef, 
                                          'pre_dy_fc1_relu', constant = False)
        self.pre_dy_fc2 = self.fc_layer(self.pre_dy_fc1_relu, 
                                        'pre_dy_fc2', 2048, bias_decay = True)
        self.pre_dy_fc2_relu = self.lrelu(self.pre_dy_fc2, 1.5 * self.relu_coef, 
                                          'pre_dy_fc2_relu', constant = False)
        self.dy_param = self.fc_layer(self.pre_dy_fc2_relu,
                                      'dy_param', self.output_feature_dim, bias_decay = True)

        self.output = tf.Variable(initial_value = 1.0, trainable = False, 
                                  validate_shape = False, dtype = tf.float32)
        self.get_output = tf.assign(self.output, self.dy_param, 
                                    validate_shape = False)

        #gather weight decays
        self.wd = tf.add_n(tf.get_collection('txt_net_weight_decay'), 
                           name = 'txt_net_total_weight_decay')
    
    def accumulate(self):
        #gradients calculation
        self.ys = [self.wd, self.dy_param]
        self.grad_ys = [1.0, self.output_grad]

        self.gradients = tf.gradients(self.ys, self.varlist, grad_ys = self.grad_ys)
        self.gradients_relu = tf.gradients(self.ys, self.varlist_relu, grad_ys = self.grad_ys)

        self.grad_and_vars = []
        self.grad_and_vars_relu = []
        
        for idx, var in enumerate(self.varlist):
            self.grad_and_vars.append((tf.clip_by_value(self.gradients[idx], -10, 10), var))
        for idx, var in enumerate(self.varlist_relu):
            self.grad_and_vars_relu.append((self.gradients_relu[idx], var))

        with tf.control_dependencies(self.gradients + self.gradients_relu):
            self.train_op = tf.group(self.opt.apply_gradients(self.grad_and_vars),
                                     self.opt_relu.apply_gradients(self.grad_and_vars_relu))
        self.safe_ops = {}
        for v in self.varlist:
            self.safe_ops[v] = tf.assign(v, tf.where(tf.is_finite(v), v, 1e-25 * tf.ones_like(v)))

    def set_input(self, texts):
        self.p_texts = texts

    def get_output(self):
        return self.p_dy_param

    def forward(self, physical_output = False):
        if physical_output:
            [self.p_dy_param] = self.sess.run([self.get_output], 
                                              feed_dict = {self.texts: 
                                                           self.p_texts})
        else:
            self.sess.run([self.get_output],
                          feed_dict = {self.texts: self.p_texts})

        return
    
    def backward(self):
        self.sess.run(self.train_op, feed_dict = {self.texts: self.p_texts})
        return

    #shape: [h, w, in_channel, out_channel]
    def conv_layer(self, bottom, name, shape,
    			   strides = [1, 1, 1, 1], weight_init_std = 0.1, bias_init_value = 0.01):
        conv_filter = self.get_weight(name, shape, weight_init_std)
        biases = self.get_bias(name, shape[3], bias_init_value)
        conv = tf.nn.conv2d(bottom, conv_filter, strides, 'SAME')
        result = tf.nn.bias_add(conv, biases)
        return result

    def fc_layer(self, bottom, name, output_shape, weight_init_std = 0.001, bias_init_value = 0.0, bias_decay = False):
        weights = self.get_weight(name, [bottom.get_shape()[1], output_shape], weight_init_std)
        biases = self.get_bias(name, [output_shape], bias_init_value, bias_decay)
        fc = tf.nn.bias_add(tf.matmul(bottom, weights), biases)
        return fc 

    def lrelu(self, x, leak = 0.1, name = 'lrelu', constant = True):
        if not constant:
            if self.data_dict.get(name) is not None:
                init = tf.constant_initializer(
                    value = self.data_dict[name], dtype = tf.float32)
            else:
                init = tf.constant_initializer(leak)
            x_shape = x.get_shape().as_list()
            x_shape[0] = 1
            with tf.variable_scope(name):
                leak = tf.get_variable(name = 'relu_params', initializer = init,
                                    shape = x_shape,  dtype = tf.float32)
                self.varlist_relu.append(leak)
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

    def get_bias(self, name, shape, init_value = 0.0, weight_decay = False):
        if self.data_dict.get(name):
            init = tf.constant_initializer(
                value = self.data_dict[name]['biases'], dtype = tf.float32)
        else:
            init = tf.constant_initializer(init_value)
            print('[WARNING]This is random init!')
        with tf.variable_scope(name):
            var = tf.get_variable(name = 'bias', initializer = init, 
                                  shape = shape, dtype = tf.float32)
            weight_decay = tf.multiply(tf.nn.l2_loss(var), self.weight_decay, 
                                       name = 'weight_decay')
        tf.add_to_collection('txt_net_weight_decay', weight_decay)
        self.varlist.append(var)
        return var

    def get_weight(self, name, shape, init_std = 0.01):
        if self.data_dict.get(name):
            init = tf.constant_initializer(
                value = self.data_dict[name]['weights'], dtype = tf.float32)
        else:
            init = tf.random_normal_initializer(mean = 0.0, stddev = init_std)
            print('[WARNING]This is random init!')
        with tf.variable_scope(name):
            var = tf.get_variable(name = 'weights', initializer = init, 
                                  shape = shape, dtype = tf.float32)
            weight_decay = tf.multiply(tf.nn.l2_loss(var), self.weight_decay, 
                                       name = 'weight_decay')
        tf.add_to_collection('txt_net_weight_decay', weight_decay)
        self.varlist.append(var)
        return var
