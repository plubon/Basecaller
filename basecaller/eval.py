from config import ConfigReader
import os
from data import EvalDataExtractor
import tensorflow as tf
from model import ModelFactory
import numpy as np
import sys


def evaluate(model_dir, data_dir, out_dir, file_list=None):
    config = ConfigReader(os.path.join(model_dir, 'config.json')).read()
    data_extractor = EvalDataExtractor(data_dir, file_list)
    dataset_handle = tf.placeholder(tf.string, shape=[])
    feedable_iterator = tf.data.Iterator.from_string_handle(dataset_handle, data_extractor.output_types,
                                                            data_extractor.output_shapes)
    signal, filename, index = feedable_iterator.get_next()
    model = ModelFactory.get(config.model_name, signal, config)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, os.path.join(model_dir, "model.ckpt"))
        print('Model restored')
        while data_extractor.has_next_file():
            logits_list = []
            filenames_list = []
            indices_list = []
            eval_dataset = data_extractor.get_next_file_dataset()
            eval_iterator = eval_dataset.make_one_shot_iterator()
            eval_handle = sess.run(eval_iterator.string_handle())
            while True:
                try:
                    logits, filenames, indices = sess.run([model.logits, filename, index],
                                                          feed_dict={dataset_handle: eval_handle})
                    filenames_list.append(filenames)
                    logits_list.append(logits)
                    indices_list.append(indices)
                except tf.errors.OutOfRangeError:
                    break
            file_logits = np.concatenate(logits_list)
            file_filenames = np.concatenate(filenames_list)
            file_indices = np.concatenate(indices_list)
            save_file_results(out_dir, file_logits, file_filenames, file_indices)


def save_file_results(out_dir, logits, filenames, indices):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    filename = os.path.join(out_dir, f"{filenames[0]}.tfrecords")
    with tf.python_io.TFRecordWriter(filename) as writer:
        logits_feature = tf.train.FloatList(value=logits)
        logits_feature = tf.train.Feature(float_list=logits_feature)
        filenames_feature = tf.train.BytesList(value=filenames)
        filenames_feature = tf.train.Feature(bytes_list=filenames_feature)
        indices_feature = tf.train.Int64List(value=indices)
        indices_feature = tf.train.Feature(int64_list=indices_feature)
        feature_dict = {
            'logits': logits_feature,
            'filenames': filenames_feature,
            'indices': indices_feature

        }
        record = tf.train.Features(feature=feature_dict)
        record = tf.train.Example(features=record)
        writer.write(record.SerializeToString())

if __name__ == "__main__":
    evaluate(sys.argv[1], sys.argv[2], sys.argv[3],
             ['DEAMERNANOPORE_20161117_FNFAB43577_MN16450_sequencing_run_MA_821_R9_4_NA12878_11_17_16_88738_ch101_read1454_strand'])