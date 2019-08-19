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


def log_to_file(path, line):
    with open(path, 'a+') as log_file:
        log_file.write(line)
        log_file.write('\n')


def train(config_path, dataset_path, output_path):
    copyfile(config_path, os.path.join(output_path, 'config.json'))
    log_path = os.path.join(output_path, 'log')
    config = ConfigReader(config_path).read()
    dataset_extractor = DatasetExtractor(dataset_path, config)
    dataset_train, train_size = dataset_extractor.extract_train()
    train_iterator = dataset_train.make_one_shot_iterator()
    dataset_val, val_size = dataset_extractor.extract_val()
    val_iterator = dataset_val.make_initializable_iterator()
    dataset_test, test_size = dataset_extractor.extract_train()
    test_iterator = dataset_test.make_one_shot_iterator()
    dataset_handle = tf.placeholder(tf.string, shape=[])
    feedable_iterator = tf.data.Iterator.from_string_handle(dataset_handle, dataset_train.output_types,
                                                            dataset_train.output_shapes, dataset_train.output_classes)
    signal, label, signal_len, _ = feedable_iterator.get_next()
    label = tf.cast(label, dtype=tf.int32)
    model = ModelFactory.get(config.model_name, signal)
    optimizer = OptimizerFactory.get(config.optimizer, model.logits, label, signal_len)
    decoder = DecoderFactory.get(config.decoder, model.logits, signal_len)
    distance_op = tf.reduce_mean(tf.edit_distance(tf.cast(decoder.decoded, dtype=tf.int32), label))
    check_op = tf.add_check_numerics_ops()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        training_handle = sess.run(train_iterator.string_handle())
        validation_handle = sess.run(val_iterator.string_handle())
        test_handle = sess.run(test_iterator.string_handle())
        epoch = 0
        steps = 0
        previous_print_length = 0
        while True:
            losses = []
            try:
                loss_value, _ = sess.run([optimizer.loss, optimizer.optimizer, check_op],
                                         feed_dict={dataset_handle: training_handle})
                losses.append(loss_value)
                steps += config.batch_size
                if previous_print_length > 0:
                    print('\b' * previous_print_length, end='', flush=True)
                message = f"Epoch: {epoch} Step: {steps} Loss: {np.mean(losses)}"
                log_to_file(log_path, message)
                previous_print_length = len(message)
                print(message, end='', flush=True)
                if steps >= train_size:
                    saver.save(sess, os.path.join(output_path, f"model.ckpt"))
                    distances = []
                    val_losses = []
                    sess.run(val_iterator.initializer)
                    while True:
                        try:
                            distance, val_loss = sess.run([distance_op, optimizer.loss, check_op],
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
            except tf.errors.OutOfRangeError:
                break  # End of dataset
        saver.save(sess, os.path.join(output_path, "model.ckpt"))
        test_distances = []
        test_losses = []
        while True:
            try:
                test_distance, test_loss = sess.run([distance_op, optimizer.loss, check_op],
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
          sys.argv[3])
