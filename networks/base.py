import os
from glob import glob
import tensorflow as tf

class Model(object):
  """Abstract object representing an Reader model."""
  def __init__(self):
    return

  def save(self, sess, checkpoint_dir, dataset_name):
    self.saver = tf.train.Saver()

    print(" [*] Saving checkpoints...")
    model_name = type(self).__name__ or "Reader"
    model_dir = dataset_name

    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)
    self.saver.save(sess, os.path.join(checkpoint_dir, model_name))

  def load(self, sess, checkpoint_dir, dataset_name, load_var):
	
    all_vars = tf.global_variables()
    if len(load_var) == 0:
      restore_vars = all_vars
    else:
      restore_vars = [var for var in all_vars if var in load_var]
    self.saver = tf.train.Saver(restore_vars)
    print(" [*] Loading checkpoints...")
    print(dataset_name)
    model_dir = dataset_name
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
      return True
    else:
      return False

