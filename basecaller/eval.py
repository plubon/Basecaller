from config import ConfigReader
import os
from data import EvalDataExtractor
import tensorflow as tf
from model import ModelFactory
import numpy as np
import sys
import json


def evaluate(model_dir, data_dir, out_dir, file_list=None):
    config = ConfigReader(os.path.join(model_dir, 'config.json')).read()
    data_extractor = EvalDataExtractor(data_dir, file_list)
    dataset_handle = tf.placeholder(tf.string, shape=[])
    feedable_iterator = tf.data.Iterator.from_string_handle(dataset_handle, data_extractor.output_types,
                                                            data_extractor.output_shapes)
    signal, index, filename = feedable_iterator.get_next()
    model = ModelFactory.get(config.model_name, signal, config)
    saver = tf.train.Saver()
    current_file = None
    with tf.Session() as sess:
        saver.restore(sess, os.path.join(model_dir, "model.ckpt"))
        print('Model restored')
        logits_list = []
        filenames_list = []
        indices_list = []
        eval_dataset = data_extractor.get_dataset()
        eval_iterator = eval_dataset.make_one_shot_iterator()
        eval_handle = sess.run(eval_iterator.string_handle())
        while True:
            try:
                logits, indices, filenames = sess.run([model.logits, index, filename],
                                                      feed_dict={dataset_handle: eval_handle})
                i = 0
                while i < filenames.shape[0]:
                    i_filename = filenames[i].decode('utf-8')
                    if current_file is None:
                        current_file = i_filename
                    if current_file != i_filename:
                        file_logits = np.stack(logits_list)
                        file_filenames = np.stack(filenames_list)
                        file_indices = np.stack(indices_list)
                        save_file_results(out_dir, file_logits, file_filenames, file_indices)
                        logits_list = []
                        filenames_list = []
                        indices_list = []
                        print(f"Processed file :{current_file}")
                        current_file = i_filename
                    filenames_list.append(filenames[i])
                    logits_list.append(logits[i, :, :])
                    indices_list.append(indices[i])
                    i = i + 1
            except tf.errors.OutOfRangeError:
                break
        print(f"Processed file :{filenames[0].decode('utf-8')}")
        file_logits = np.stack(logits_list)
        file_filenames = np.stack(filenames_list)
        file_indices = np.stack(indices_list)
        save_file_results(out_dir, file_logits, file_filenames, file_indices)


def save_file_results(out_dir, logits, filenames, indices):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    file_path = os.path.join(out_dir, f"{filenames[0].decode('utf-8')}")
    np.save(file_path, logits)


if __name__ == "__main__":
    with open(sys.argv[4]) as file_list:
        data = json.loads(file_list.read())
        lines = data['files']
    evaluate(sys.argv[1], sys.argv[2], sys.argv[3], lines)