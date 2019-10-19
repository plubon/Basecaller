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
    dataset_test, test_size = dataset_extractor.extract_train()
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
    test_handle = sess.run(test_iterator.string_handle())
    test_distances = []
    test_losses = []
    while True:
        try:
            test_distance, test_loss = sess.run([distance_op, optimizer.loss],
                                          feed_dict={dataset_handle: test_handle})
            print(test_distance)
            test_distances.append(test_distance)
            test_losses.append(test_loss)
        except tf.errors.OutOfRangeError:
            break
    mean_test_distance = np.mean(test_distances)
    mean_test_loss = np.mean(test_losses)
    print(flush=True)
    log_message = f"Test Loss: {mean_test_loss} Edit Distance: {mean_test_distance}"
    print(log_message, flush=True)
    log_to_file(log_path, log_message)


if __name__ == "__main__":
    test(sys.argv[1],
          sys.argv[2])
