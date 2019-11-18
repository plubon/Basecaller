import tensorflow as tf
from config import ConfigReader
from data import DatasetExtractor
from model import ModelFactory
from optimizer import OptimizerFactory
from decoder import DecoderFactory
import numpy as np
import os
from shutil import copyfile
import sys
from tensorflow.python import debug as tf_debug
from utils import  log_to_file



def test(model_path, dataset_path):
    log_path = os.path.join(model_path, 'test_log')
    config = ConfigReader(os.path.join(model_path, 'config.json')).read()
    dataset_extractor = DatasetExtractor(dataset_path, config)
    dataset_test, test_size = dataset_extractor.extract()
    test_iterator = dataset_test.make_one_shot_iterator()
    dataset_handle = tf.placeholder(tf.string, shape=[])
    feedable_iterator = tf.data.Iterator.from_string_handle(dataset_handle, dataset_test.output_types,
                                                            dataset_test.output_shapes, dataset_test.output_classes)
    signal, label, signal_len, _ = feedable_iterator.get_next()
    label = tf.cast(label, dtype=tf.int32)
    model = ModelFactory.get(config.model_name, signal, config)
    optimizer = OptimizerFactory.get(config.optimizer, model.logits, label, signal_len)
    decoder = DecoderFactory.get(config.decoder, model.logits, signal_len)
    distance_op = tf.reduce_mean(tf.edit_distance(tf.cast(decoder.decoded, dtype=tf.int32), label))
    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, os.path.join(model_path, "model.ckpt"))
    tf.saved_model.simple_save(sess,
                               os.path.join(model_path, 'saved_model'),
                               inputs={"signal": signal,
                                       "lengths": signal_len},
                               outputs={"logits": model.logits})



if __name__ == "__main__":
    test(sys.argv[1],
          sys.argv[2])
