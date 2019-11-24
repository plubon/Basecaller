import os
from data import EvalDataExtractor
import tensorflow as tf
import numpy as np
import sys
import json
batch_size = 150


def evaluate(model_dir, data_dir, out_dir, file_list=None):
    data_extractor = EvalDataExtractor(data_dir, file_list)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        tf.saved_model.loader.load(sess, ["serve"], os.path.join(model_dir, 'saved_model'))
        i = 1
        size = data_extractor.get_size()
        while data_extractor.has_next_file():
            logits_list = []
            signal, lengths, indices, filename = data_extractor.extract_next_file()
            print(f"{i}/{size} : {filename} ")
            i = i + 1
            if len(signal) == 0:
                continue
            batch_index = 0
            while batch_index < signal.shape[0]:
                feed_dict = {
                    'IteratorGetNext:0': signal[batch_index:batch_index + batch_size, :, :],
                    'IteratorGetNext:2': lengths[batch_index:batch_index + batch_size]
                }
                logits = sess.run('dense/BiasAdd:0', feed_dict=feed_dict)
                logits_list.append(logits)
                batch_index = batch_index + batch_size
            file_logits = np.concatenate(logits_list, axis=0)
            save_file_results(out_dir, file_logits, filename, indices)
            print(f"Processed file :{filename} ")


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
