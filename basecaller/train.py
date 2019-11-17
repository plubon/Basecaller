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
from utils import log_to_file

def train(config_path, train_dataset_path, val_dataset_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    copyfile(config_path, os.path.join(output_path, 'config.json'))
    log_path = os.path.join(output_path, 'log')
    config = ConfigReader(config_path).read()
    train_dataset_extractor = DatasetExtractor(train_dataset_path, config)
    val_dataset_extractor = DatasetExtractor(val_dataset_path, config)
    dataset_train, train_size = train_dataset_extractor.extract()
    train_iterator = dataset_train.make_one_shot_iterator()
    dataset_val, val_size = val_dataset_extractor.extract()
    dataset_test = dataset_val.take(300)
    dataset_val = dataset_val.take(75)
    val_iterator = dataset_val.make_initializable_iterator()
    test_iterator = dataset_test.make_one_shot_iterator()
    dataset_handle = tf.placeholder(tf.string, shape=[])
    feedable_iterator = tf.data.Iterator.from_string_handle(dataset_handle, dataset_train.output_types,
                                                            dataset_train.output_shapes, dataset_train.output_classes)
    signal, label, signal_len, _ = feedable_iterator.get_next()
    label = tf.cast(label, dtype=tf.int32)
    model = ModelFactory.get(config.model_name, signal, config)
    optimizer = OptimizerFactory.get(config.optimizer, model.logits, label, signal_len)
    decoder = DecoderFactory.get(config.decoder, model.logits, signal_len)
    distance_op = tf.reduce_mean(tf.edit_distance(tf.cast(decoder.decoded, dtype=tf.int32), label))
    saver = tf.train.Saver()
    sess = tf.Session()
    if config.debug:
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    sess.run(tf.global_variables_initializer())
    training_handle = sess.run(train_iterator.string_handle())
    validation_handle = sess.run(val_iterator.string_handle())
    test_handle = sess.run(test_iterator.string_handle())
    epoch = 0
    steps = 0
    previous_print_length = 0
    losses = []
    while True:
        try:
            loss_value, _ = sess.run([optimizer.loss, optimizer.optimizer],
                                     feed_dict={dataset_handle: training_handle})
            losses.append(loss_value)
            steps += config.batch_size
            if previous_print_length > 0:
                print('\b' * previous_print_length, end='', flush=True)
            message = f"Epoch: {epoch} Step: {steps} Step Loss:{loss_value} Epoch Loss: {np.mean(losses)}"
            log_to_file(log_path, message)
            previous_print_length = len(message)
            print(message, end='', flush=True)
            if steps >= train_size:
                saver.save(sess, os.path.join(output_path, f"model.ckpt"))
                if config.validate:
                    distances = []
                    val_losses = []
                    sess.run(val_iterator.initializer)
                    while True:
                        try:
                            distance, val_loss = sess.run([distance_op, optimizer.loss],
                                                          feed_dict={dataset_handle: validation_handle})
                            distances.append(distance)
                            val_losses.append(val_loss)
                        except tf.errors.InvalidArgumentError as e:
                            log_to_file(log_path, e.message)
                            raise e
                        except tf.errors.OutOfRangeError:
                            break
                    mean_distance = np.mean(distances)
                    mean_val_loss = np.mean(val_losses)
                    print(flush=True)
                    log_message = f"Epoch: {epoch} Validation Loss: {mean_val_loss} Edit Distance: {mean_distance}"
                    print(log_message, flush=True)
                    log_to_file(log_path, log_message)
                epoch += 1
                steps = 0
                previous_print_length = 0
                losses = []
        except tf.errors.OutOfRangeError:
            break  # End of dataset
    saver.save(sess, os.path.join(output_path, "model.ckpt"))
    test_distances = []
    test_losses = []
    while True:
        try:
            test_distance, test_loss = sess.run([distance_op, optimizer.loss],
                                          feed_dict={dataset_handle: test_handle})
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
    train(sys.argv[1],
          sys.argv[2],
          sys.argv[3],
          sys.argv[4])
