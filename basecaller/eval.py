from config import ConfigReader
import os
from data import EvalDataExtractor
import tensorflow as tf
from model import ModelFactory
import numpy as np


def evaluate(model_dir, data_dir, file_list=None):
    config = ConfigReader(os.path.join(model_dir, 'config.json')).read()
    dataset_extractor = EvalDataExtractor(data_dir, file_list)
    dataset_eval, eval_size = dataset_extractor.extract_for_eval('test')
    eval_iterator = dataset_eval.make_one_shot_iterator()
    dataset_handle = tf.placeholder(tf.string, shape=[])
    feedable_iterator = tf.data.Iterator.from_string_handle(dataset_handle, dataset_eval.output_types,
                                                            dataset_eval.output_shapes, dataset_eval.output_classes)
    signal, filename, index = feedable_iterator.get_next()
    model = ModelFactory.get(config.model_name, signal, config)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        while True:
            try:
                saver.restore(sess, os.path.join(model_dir, "model.ckpt"))
                logits, filename, index = sess.run([model.logits, filename, index],
                                                   feed_dict={dataset_handle: eval_iterator})
                #save logits
            except tf.errors.OutOfRangeError:
                break

