#!/usr/bin/python3
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from tensorflow.core.protobuf import config_pb2
import json
import os
from config import FLAGS
from networks.net_wrapper import NetWrapper
from data_loader import DataLoader
from test import test
from utils import val_ids, test_ids, visualize

import pdb

def step(net, loader):
    batch_data = loader.get_batch()
    net.set_input(batch_data)
    net.forward_backward()

def main(_):
    sess = tf.Session(config = tf.ConfigProto(allow_soft_placement = True, 
                                              log_device_placement = False))

    #declare networks
    with tf.device('/gpu: %d' % FLAGS.DEVICE_NUM):
        net = NetWrapper(sess, FLAGS.IMAGE_MODEL, FLAGS.image_lr_conv, FLAGS.image_lr_region, FLAGS.text_lr,
        				 FLAGS.pair_net_batch_size, FLAGS.MODE,
        				 FLAGS.IMAGE_FINE_TUNE_MODEL, FLAGS.TEXT_FINE_TUNE_MODEL)
        net.build()

    if  FLAGS.DEBUG:
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

    net.text_net.sess = sess
    init = tf.global_variables_initializer()
    sess.run(init)

    #train_writer = tf.summary.FileWriter('.' + '/train', sess.graph)
    #restore network
    if FLAGS.RESTORE_ALL:
        restore = []
    else:
        restore = net.varlist

    if net.load(sess, FLAGS.SNAPSHOT_DIR, 'nldet_%s_%d' % (FLAGS.INIT_SNAPSHOT, FLAGS.INIT_ITER), restore):
        print('[INIT]Successfully load model from %s_%d' % (FLAGS.INIT_SNAPSHOT, FLAGS.INIT_ITER))
    elif FLAGS.MODE == 'test':
        print('[INIT]No Tensorflow found  model for %s, test initialization' % FLAGS.INIT_SNAPSHOT)
    else:
        print('[INIT]No Tensorflow found  model for %s train from scratch' % FLAGS.INIT_SNAPSHOT)

    if FLAGS.MODE == "train":
        resume_status = None
        status_dir = '%s/nldet_%s_%d/nldet_status_%s_%d.json' %\
            (FLAGS.SNAPSHOT_DIR, FLAGS.INIT_SNAPSHOT, FLAGS.INIT_ITER, FLAGS.INIT_SNAPSHOT, FLAGS.INIT_ITER)
        if os.path.exists(status_dir):
            resume_status = json.load(open(status_dir, 'r'))
            print('resume from %s' % status_dir)
        else:
            print('no resume data loader status found')

        # initialize data loader
        if resume_status is None:
            loader = DataLoader(FLAGS.NUM_PROCESSORS, FLAGS.batch_size, FLAGS.MODE, capacity = FLAGS.DATA_LOADER_CAPACITY)
        else:
            loader = DataLoader(FLAGS.NUM_PROCESSORS, FLAGS.batch_size, FLAGS.MODE,
                                resume_status['batch_idx'], resume_status['data_ids'], FLAGS.DATA_LOADER_CAPACITY)
        loader.start()

        current_iter = FLAGS.INIT_ITER + 1
        while current_iter <= FLAGS.MAX_ITERS:
            step(net, loader)
            if current_iter % FLAGS.PRINT_ITERS == 0:
                net.get_output(current_iter)
            if current_iter % FLAGS.SAVE_ITERS == 0:
                net.save(sess, FLAGS.SNAPSHOT_DIR, 'nldet_%s_%d' % (FLAGS.PHASE, current_iter))
                saving_status = loader.get_status()
                json.dump(saving_status, open('%s/nldet_%s_%d/nldet_status_%s_%d.json' % \
                    (FLAGS.SNAPSHOT_DIR, FLAGS.PHASE, current_iter, FLAGS.PHASE, current_iter), 'w'))
                print('save data loader status to nldet_status_%s_%d.json' % (FLAGS.PHASE, current_iter))
            current_iter += 1
        loader.stop()
    
    elif FLAGS.MODE == "test" or FLAGS.MODE == 'val':
        if FLAGS.MODE == 'test':
            tranverse_ids = test_ids
        else:
            tranverse_ids = val_ids
            if FLAGS.LEVEL != 'level_0':
                print('Validation set only support level-0')
                return
        for idx, tid in enumerate(tranverse_ids):
            print('[%d/%d]' % (idx + 1, len(tranverse_ids)))
            result_dir = "nlvd_evaluation/results/vg_v1/dbnet_%s" % FLAGS.IMAGE_MODEL
            if os.path.exists('%s/tmp_output/%s_%d.txt' % (result_dir, FLAGS.LEVEL, tid)):
                print('FOUND EXISTING RESULT')
                continue
            test(net, tid, FLAGS.LEVEL, result_dir, top_num = FLAGS.TOP_NUM_RPN, gt_box = FLAGS.INCLUDE_GT_BOX)
        os.system('cat %s/tmp_output/%s* > %s/%s.txt' % (result_dir, FLAGS.LEVEL, result_dir, FLAGS.LEVEL))
    
    elif FLAGS.MODE == "vis":
        im_ids = json.load(open(FLAGS.IMAGE_LIST_DIR, 'r')) 
        os.makedirs(FLAGS.VIS_DIR, exist_ok = True)
        for idx, im_id in enumerate(im_ids):
            detection_result = test(net, im_id, 'vis', None,
                top_num = FLAGS.TOP_NUM_RPN, query_phrase = [FLAGS.PHRASE_INPUT])
            visualize(im_id, detection_result, FLAGS.VIS_NUM, FLAGS.PHRASE_INPUT,
                      os.path.join(FLAGS.VIS_DIR, 'vis_' + str(idx + 1)))
    return

if __name__ == '__main__':
    tf.app.run()
