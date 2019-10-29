from config import ConfigReader
import os
from data import EvalDataExtractor
import tensorflow as tf
from model import ModelFactory
from Levenshtein import distance
import numpy as np
import sys
import json
from utils import int_label_to_string
batch_size = 150


def evaluate(model_dir, data_dir, out_dir, file_list=None):
    data_extractor = EvalDataExtractor(data_dir, file_list)
    logits_input = tf.placeholder(tf.float32, shape=(None, 300, 5), name='decode_logits_input')
    len_input = tf.placeholder(tf.int32, shape=(None,), name='decode_lengths_input')
    decoded = tf.sparse.to_dense(tf.nn.ctc_beam_search_decoder(tf.transpose(logits_input, perm=[1, 0, 2]),
                                  tf.cast(len_input, tf.int32 , name='aaa2'), merge_repeated=False)[0][0],
                                 default_value=-1, name='s_to_d')
    file_results = {}
    with tf.Session() as sess:
        tf.saved_model.loader.load(sess, ["serve"], os.path.join(model_dir, 'saved_model'))
        while data_extractor.has_next_file():
            print(f"Processing file..")
            logits_list = []
            signal, lengths, label, indices, filename = data_extractor.extract_next_file()
            batch_index = 0
            distances = []
            while batch_index < signal.shape[0]:
                feed_dict = {
                    'IteratorGetNext:0': signal[batch_index:batch_index + batch_size, :, :],
                    'IteratorGetNext:2': lengths[batch_index:batch_index + batch_size]
                }
                logits = sess.run('dense/BiasAdd:0', feed_dict=feed_dict)
                logits_list.append(logits)
                decoded_out = sess.run(decoded, feed_dict={logits_input: logits, len_input: lengths[batch_index:batch_index + batch_size]})
                for indx, bpread in enumerate(decoded_out):
                    if np.any(bpread == -1):
                        bpread = bpread[:np.argmax(bpread == -1)]
                    if len(bpread) == 0:
                        continue
                    string_result = int_label_to_string(bpread)
                    str_target = [x.decode("utf-8") for x in label[indx]]
                    joined_target = ''.join(str_target)
                    dist = distance(string_result, joined_target) / len(joined_target)
                    distances.append(dist)
                batch_index = batch_index + batch_size
            file_logits = np.concatenate(logits_list, axis=0)
            save_file_results(out_dir, file_logits, filename, indices)
            file_results[filename] = np.mean(distances)
            print(f"Processed file :{filename} Distance: {file_results[filename]}")
    with open(os.path.join(out_dir, 'file_results.csv'),'w') as results_file:
        for key, value in file_results.items():
            results_file.write(f"{key},{value}\n")


def save_file_results(out_dir, logits, filename, indices):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    file_path = os.path.join(out_dir, filename)
    np.save(file_path, logits)


if __name__ == "__main__":
    with open(sys.argv[4]) as file_list:
        data = json.loads(file_list.read())
        lines = data['files']
    evaluate(sys.argv[1], sys.argv[2], sys.argv[3], lines)
