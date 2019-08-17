import tensorflow as tf
from config import ConfigReader
from data import DatasetExtractor
from model import ModelFactory
from optimizer import OptimizerFactory
from decoder import DecoderFactory
import numpy as np


def train(config_path, dataset_path, output_path):
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
                                                            dataset_train.output_shapes)
    signal, label, signal_len, _ = feedable_iterator.get_next()
    label = tf.cast(label, dtype=tf.int32)
    model = ModelFactory.get(config.model_name, signal)
    optimizer = OptimizerFactory.get(config.optimizer, model.logits, label, signal_len)
    decoder = DecoderFactory.get(config.predictor, model.logits, signal_len)
    distance = tf.reduce_mean(tf.edit_distance(tf.cast(decoder.decoded, dtype=tf.int32), label))
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
                loss_value, _ = sess.run([optimizer.loss, optimizer.optimizer],
                                         feed_dict={dataset_handle: training_handle})
                losses.append(loss_value)
                steps += config.batch_size
                if previous_print_length > 0:
                    print('\b' * previous_print_length, end='')
                message = f"Epoch: {epoch} Step: {steps} Loss: {np.mean(losses)}"
                previous_print_length = len(message)
                print(message, end='')
                if steps >= train_size:
                    distances = []
                    val_losses = []
                    while True:
                        try:
                            distance, val_loss = sess.run([distance, optimizer.loss],
                                                          feed_dict={dataset_handle: validation_handle})
                            distances.append(distance)
                            val_losses.append(val_loss)
                        except tf.errors.OutOfRangeError:
                            break
                    mean_distance = np.mean(distances)
                    mean_val_loss = np.mean(val_losses)
                    print()
                    print(f"Epoch: {epoch} Validation Loss: {mean_val_loss} Edit Distance: {mean_distance}")
                    epoch += 1
                    steps = 0
                    previous_print_length = 0
            except tf.errors.OutOfRangeError:
                break  # End of dataset
        test_distances = []
        test_losses = []
        while True:
            try:
                test_distance, test_loss = sess.run([distance, optimizer.loss],
                                              feed_dict={dataset_handle: test_handle})
                test_distances.append(test_distance)
                test_losses.append(test_loss)
            except tf.errors.OutOfRangeError:
                break
        mean_test_distance = np.mean(test_distances)
        mean_test_loss = np.mean(test_losses)
        print()
        print(f"Test Loss: {mean_test_loss} Edit Distance: {mean_test_distance}")

if __name__ == "__main__":
    train("/home/piotr/Uczelnia/PracaMagisterska/Basecaller/basecaller/configs/test.json",
          '/home/piotr/Uczelnia/PracaMagisterska/Dane/dataset_chiron', None)
