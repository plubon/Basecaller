from utils import string_label_to_int, write_dict_to_file
from sklearn.model_selection import train_test_split
from entities import TrainingExample
from reader import ChironFileReader, H5FileReader, DirtyChironFileReader, OverlapChironFileReader
import os
import tensorflow as tf


class BaseFileParser:

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

    def create_record(self):
        count = 0
        test_filename = os.path.join(self.output_path, 'dataset.tfrecords')
        with tf.python_io.TFRecordWriter(test_filename) as writer:
            i = 1
            for file in self.files:
                print(f"{i}/{len(self.files)}", flush=True)
                examples = self.reader.read(os.path.join(self.input_path, file))
                count += len(examples)
                i = i + 1
                for example in examples:
                    writer.write(self.get_tf_example(example).SerializeToString())
        info_filename = os.path.join(self.output_path, 'info.json')
        write_dict_to_file(info_filename, {
            'count': count,
            'files': self.files
        })

    def create(self, split=False):
        if split:
            train, rest = train_test_split(self.files, train_size=1 - self.test_size - self.val_size)
            val, test = train_test_split(rest, train_size=self.val_size / (self.val_size + self.test_size))
            self.create_split_record(train, 'train')
            self.create_split_record(test, 'test')
            self.create_split_record(val, 'val')
        else:
            self.create_record()


class CleanFileParser(BaseFileParser):

    def __init__(self, input_path, output_path, seg_length=300, skip=10, format='h5', test_size=0.2, val_size=0.2):
        super().__init__(input_path, output_path, seg_length, skip, format, test_size, val_size)

    known_formats = ['h5', 'chiron']

    def get_reader(self, format):
        if format == 'h5':
            return H5FileReader(self)
        elif format == 'chiron':
            return ChironFileReader(self)
        else:
            raise ValueError(f"Format was {format}, but it must be one of {', '.join(self.known_formats)}.")


class DirtyFileParser(BaseFileParser):

    def __init__(self, input_path, output_path, seg_length=300, skip=10, format='h5', test_size=0.2, val_size=0.2):
        super().__init__(input_path, output_path, seg_length, skip, format, test_size, val_size)

    known_formats = ['h5', 'chiron']

    def get_reader(self, format):
        if format == 'h5':
            return H5FileReader(self)
        elif format == 'chiron':
            return DirtyChironFileReader(self)
        else:
            raise ValueError(f"Format was {format}, but it must be one of {', '.join(self.known_formats)}.")

class OverlapFileParser(BaseFileParser):

    def __init__(self, input_path, output_path, seg_length=300, skip=10, format='h5', test_size=0.2, val_size=0.2):
        super().__init__(input_path, output_path, seg_length, skip, format, test_size, val_size)

    known_formats = ['h5', 'chiron']

    def get_reader(self, format):
        if format == 'h5':
            return H5FileReader(self)
        elif format == 'chiron':
            return OverlapChironFileReader(self)
        else:
            raise ValueError(f"Format was {format}, but it must be one of {', '.join(self.known_formats)}.")