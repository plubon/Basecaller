from entities import TrainingExample
from reader import ChironFileReader, H5FileReader
import numpy as np
import os
import tensorflow as tf
import sys
import json
from parser import CleanFileParser, DirtyFileParser, OverlapFileParser


class DatasetExtractor:
    train_features = {
        'signal': tf.FixedLenFeature([300], tf.float32),
        'label': tf.VarLenFeature(tf.int64),
        'signal_len': tf.FixedLenFeature([], tf.int64),
        'label_len': tf.FixedLenFeature([], tf.int64)
    }

    def __init__(self, dataset_path, config):
        self.config = config
        self.path = dataset_path

    def extract_train(self):
        return self._extract('train')

    def extract(self):
        with tf.device('/cpu:0'):
            dataset = tf.data.TFRecordDataset([os.path.join(self.path, 'dataset.tfrecords')])
            dataset = dataset.shuffle(self.config.batch_size * 5)
            dataset = dataset.repeat(self.config.epochs)
            dataset = dataset.map(self._extract_fn)
            dataset = dataset.batch(self.config.batch_size)
            dataset = dataset.prefetch(self.config.batch_size * 5)
            with open(os.path.join(self.path, 'info.json'), 'r') as info_file:
                info = json.load(info_file)
                size = int(info['count'])
        return dataset, size

    def extract_test(self):
        return self._extract('test')

    def extract_val(self):
        return self._extract('val')

    def _extract_fn(self, tfrecord):
        sample = tf.parse_single_example(tfrecord, self.train_features)

        return [tf.expand_dims(sample['signal'], -1), sample['label'], sample['signal_len'], sample['label_len']]

    def _extract(self, split_name):
        with tf.device('/cpu:0'):
            dataset = tf.data.TFRecordDataset([os.path.join(self.path, f"{split_name}.tfrecords")])
            dataset = dataset.shuffle(self.config.batch_size * 5)
            dataset = dataset.repeat(self.config.epochs)
            dataset = dataset.map(self._extract_fn)
            dataset = dataset.batch(self.config.batch_size)
            dataset = dataset.prefetch(self.config.batch_size * 5)
            with open(os.path.join(self.path, split_name), 'r') as info_file:
                info = json.load(info_file)
                size = int(info['count'])
        return dataset, size


class EvalDataExtractor:

    def __init__(self, data_dir, file_list=None):
        self.data_dir = data_dir
        self.files = os.listdir(data_dir)
        self.files = [x for x in self.files if x.endswith('fast5') or x.endswith('signal')]
        if file_list is not None:
            self.files = [x for x in self.files if x.split('.')[0] in file_list]
        self.current_file = 0
        self.current_row = 0
        self.limit = None

    def set_limit(self, limit):
        self.limit = limit

    def filter(self, name):
        self.files = [x for x in self.files if name in x]

    def get_size(self):
        size = len(self.files)
        if self.limit is not None and self.limit < len(self.files):
            size = self.limit
        return size

    def has_next_file(self):
        size = len(self.files)
        if self.limit is not None and self.limit < len(self.files):
            size = self.limit
        return self.current_file < size

    def get_next_file_name(self):
        return self.files[self.current_file]

    def extract_next_file(self):
        filename = self.files[self.current_file]
        if filename.endswith('fast5'):
            reader = H5FileReader()
        else:
            reader = ChironFileReader()
        signal, index = reader.read_for_eval(os.path.join(self.data_dir, filename.split('.')[0]))
        if len(signal) == 0:
            self.current_file = self.current_file + 1
            return [], [], [], []
        signal = np.expand_dims(np.stack(signal).astype(np.float32), -1)
        index = np.array(index)
        lengths = np.repeat(300, index.shape[0])
        self.current_file = self.current_file + 1
        return signal, lengths, index, filename


if __name__ == "__main__":
    parser = OverlapFileParser(sys.argv[1], sys.argv[2], format=sys.argv[3])
    parser.create()
