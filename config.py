""" Configuration for the network
"""
import os
import os.path as osp
import sys
from easydict import EasyDict as edict
import tensorflow as tf

############################################### 
#set global configuration for network training#
############################################### 
flags = tf.app.flags
FLAGS = flags.FLAGS
#model hyperparameters
flags.DEFINE_string('IMAGE_MODEL', 'vgg16', 'which network to use for image pathway, <vgg16|resnet101>')
flags.DEFINE_float('text_lr', 1e-5, 'learning rate for the text end')
flags.DEFINE_float('image_lr_conv', 1e-4, 'learning rate for the image end')
flags.DEFINE_float('image_lr_region', 1e-4, 'learning rate for the image end')
flags.DEFINE_integer('batch_size', 2, 'number of images in a batch sent to the network')
flags.DEFINE_integer('pair_net_batch_size', 128, 'number of image and text pair sent to the pair net in a subbatch')
#device layout
flags.DEFINE_integer('DEVICE_NUM', 0, 'GPU device ID')
flags.DEFINE_integer('NUM_PROCESSORS', 4, 'Number of processor for data loading')
flags.DEFINE_integer('DATA_LOADER_CAPACITY', 10, 'Maximum number of batches saved in the data loader')
#training mode
flags.DEFINE_string('MODE', "train", 'train|test|val')
flags.DEFINE_boolean('DEBUG', False, 'whether run in tensorflow debug mode')
flags.DEFINE_string('PHASE', 'phase1', 'phase1|phase2')
flags.DEFINE_string('IMAGE_FINE_TUNE_MODEL', 'Region_Feat_Net.npy',
    'relative path to networks/image_feat_net/<vgg16|resnet101>/net.py depends on the choice of --IMAGE_MODEL')
flags.DEFINE_string('TEXT_FINE_TUNE_MODEL', 'Text_Feat_Net.npy', 'relative path to networks/text_feat_net/net.py')
flags.DEFINE_string('INIT_SNAPSHOT', 'phase1', 'init train from which phase')
flags.DEFINE_integer('INIT_ITER', 0, 'init train from which iteration, together with INIT_SNAPSHOT')
flags.DEFINE_string('SNAPSHOT_DIR', 'checkpoints', 'init train from which phase')
flags.DEFINE_boolean('RESTORE_ALL', False, 'restore model with all variables (concern with momentum issue)')
#test configs if testing
flags.DEFINE_string('LEVEL', 'level_0', 'level_0|level_1|level_2')
flags.DEFINE_integer('TOP_NUM_RPN', 500, 'doing nms in the top k boxes based on the prediction score')
flags.DEFINE_boolean('INCLUDE_GT_BOX', False, 'include ground truth box in final test box')
#visualization output
flags.DEFINE_string('VIS_DIR', 'visualization', 'save image detection example')
flags.DEFINE_string('PHRASE_INPUT', 'A man in red.', 'query phrase to do detections')
flags.DEFINE_string('IMAGE_LIST_DIR', 'image_examples.json', 'a file specify which images to do visualization, '
                    'image id if images in VISUAL GENOME, otherwise absolute directory of the images')
flags.DEFINE_integer('VIS_NUM', 3, 'draw how top-x detected regions')
#training infos
flags.DEFINE_integer('MAX_ITERS', float('inf'), 'Maxiumum running iteration')
flags.DEFINE_integer('PRINT_ITERS', 1, 'Print data each print_iters')
flags.DEFINE_integer('SAVE_ITERS', 2000, 'Frequency of saving checkpoints')

############################################### 
#  set global configuration for data reading  #
############################################### 
DATA_PATH = osp.abspath(osp.join(osp.dirname(__file__), 'data'))
ENV_PATHS = edict()

# need to be moved to data path
ENV_PATHS.IMAGE_PATH = '/mnt/brain3/datasets/VisualGenome/images'
ENV_PATHS.EDGEBOX_PATH = '/mnt/brain2/scratch/yutingzh/object-det-cache/nldet_cache/region_proposal_cache/vg/edgebox'
ENV_PATHS.EDGE_BOX_RPN = '/mnt/brain1/scratch/yuanluya/nldet_tensorflow/edge_boxes_with_python'
ENV_PATHS.RAW_DATA = osp.abspath(osp.join(DATA_PATH, 'region_description.json'))
ENV_PATHS.METEOR = osp.abspath(osp.join(DATA_PATH, 'meteor.json')) #upper triangle matrix 
ENV_PATHS.FREQUENCY = osp.abspath(osp.join(DATA_PATH, 'freq.json')) 
ENV_PATHS.SPLIT = osp.abspath(osp.join(DATA_PATH, 'densecap_splits.json'))
ENV_PATHS.LEVEL1_TEST = osp.abspath(osp.join(DATA_PATH, 'level1_im2p.json'))
ENV_PATHS.LEVEL2_TEST = osp.abspath(osp.join(DATA_PATH, 'level2_im2p.json'))

############################################### 
# set global configuration for data sampling  #
############################################### 
DS_CONFIG = edict()

DS_CONFIG.thre_neg = 0.1
DS_CONFIG.thre_pos = 0.9
DS_CONFIG.pos_loss_weight = 1
DS_CONFIG.neg_loss_weight = 1
DS_CONFIG.rest_loss_weight = 1
DS_CONFIG.meteor_thred = 0.3
DS_CONFIG.text_tensor_sequence_length = 256
DS_CONFIG.text_rand_sample_size = 100
DS_CONFIG.target_size = 600
DS_CONFIG.max_size = 1000
DS_CONFIG.edge_box_high_rank_num = 100
DS_CONFIG.edge_box_random_num = 50
