from utils import string_label_to_int, write_dict_to_file
from sklearn.model_selection import train_test_split
from entities import TrainingExample
from reader import ChironFileReader, H5FileReader
import numpy as np
import os
import tensorflow as tf
import sys
import json


class SignalFileParser:
    known_formats = ['h5', 'chiron']

    def get_reader(self, format):
        if format == 'h5':
            return H5FileReader(self)
        elif format == 'chiron':
            return ChironFileReader(self)
        else:
            raise ValueError(f"Format was {format}, but it must be one of {', '.join(self.known_formats)}.")

    def get_id(self):
        id = self.id_counter
        self.id_counter += 1
        return str(id)

    def __init__(self, input_path, output_path, seg_length=300, skip=10, format='h5', test_size=0.2, val_size=0.2):
        self.input_path = input_path
        self.output_path = output_path
        self.files = os.listdir(self.input_path)
        self.reader = self.get_reader(format)
        self.files = self.reader.filter_files(self.files)
        self.files = [x.split('.')[0] for x in self.files]
        self.files = list(set(self.files))
        self.skip = skip
        self.segment_length = seg_length
        self.id_counter = 1
        self.test_size = test_size
        self.val_size = val_size

    def get_tf_example(self, example):
        signal = tf.train.FloatList(value=example.signal)
        signal = tf.train.Feature(float_list=signal)
        label = tf.train.Int64List(value=string_label_to_int(example.sequence))
        label = tf.train.Feature(int64_list=label)
        signal_len = tf.train.Int64List(value=[len(example.signal)])
        signal_len = tf.train.Feature(int64_list=signal_len)
        label_len = tf.train.Int64List(value=[len(example.sequence)])
        label_len = tf.train.Feature(int64_list=label_len)
        feature_dict = {
            'signal': signal,
            'label': label,
            'signal_len': signal_len,
            'label_len': label_len
        }
        features = tf.train.Features(feature=feature_dict)
        return tf.train.Example(features=features)

    def create_split_record(self, files, split_name):
        count = 0
        test_filename = os.path.join(self.output_path, f'{split_name}.tfrecords')
        with tf.python_io.TFRecordWriter(test_filename) as writer:
            for file in files:
                examples = self.reader.read(os.path.join(self.input_path, file))
                count += len(examples)
                for example in examples:
                    writer.write(self.get_tf_example(example).SerializeToString())
        info_filename = os.path.join(self.output_path, split_name)
        write_dict_to_file(info_filename, {
            'count': count,
            'files': files
        })

    def create(self):
        train, rest = train_test_split(self.files, train_size=1 - self.test_size - self.val_size)
        val, test = train_test_split(rest, train_size=self.val_size / (self.val_size + self.test_size))
        self.create_split_record(train, 'train')
        self.create_split_record(test, 'test')
        self.create_split_record(val, 'val')


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

    def __init__(self, data_dir, file_list):
        self.data_dir = data_dir
        self.files = os.listdir(data_dir)
        if file_list is not None:
            self.files = [x for x in self.files if x.split('.')[0] in file_list]
        print(self.files)
        self.current_file = 0
        self.current_row = 0

    def get_size(self):
        return len(self.files)

    def has_next_file(self):
        return self.current_file < len(self.files)

    def extract_next_file(self):
        filename = self.files[self.current_file]
        if filename.endswith('fast5'):
            reader = H5FileReader()
        else:
            reader = ChironFileReader()
        signal, label, index = reader.read_for_eval(os.path.join(self.data_dir, filename.split('.')[0]))
        if len(signal) == 0:
            self.current_file = self.current_file + 1
            return [], [], [], [], []
        signal = np.expand_dims(np.stack(signal).astype(np.float32), -1)
        index = np.array(index)
        lengths = np.repeat(300, index.shape[0])
        self.current_file = self.current_file + 1
        return signal, lengths, label, index, filename


if __name__ == "__main__":
    parser = SignalFileParser(sys.argv[1], sys.argv[2], format=sys.argv[3])
    parser.create()
