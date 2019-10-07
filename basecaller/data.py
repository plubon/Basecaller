from utils import alphabet_dict, string_label_to_int, write_dict_to_file
from sklearn.model_selection import train_test_split
import h5py
import numpy as np
import os
import tensorflow as tf
import sys
import json


class TrainingExample:

    def __init__(self, id, filename, index, signal, raw_signal, sequence, breaks):
        self.id = id
        self.filename = filename
        self.index = index
        self.signal = signal
        self.sequence = sequence
        self.breaks = breaks
        self.raw_signal = raw_signal


class H5FileReader:

    def __init__(self, parser=None):
        self.parser = parser

    def read_for_eval(self, path):
        with h5py.File(path, 'r') as h5_file:
            raw_signal = h5_file['Raw/Reads']
            signal_path = raw_signal.visit(self.find_signal)
            dataset = raw_signal[signal_path][()]
            i = 0
            signal = []
            indices = []
            mean = np.mean(np.unique(dataset))
            std = np.std(np.unique(dataset))
            while i + 300 < len(dataset):
                raw = dataset[i:i + 300]
                normalized = (raw - mean) / std
                signal.append(normalized)
                indices.append(i)
                i = i + 30
            return signal, indices

    def filter_files(self, files):
        return [x for x in files if x.endswith('fast5')]

    def find_signal(self, name):
        if 'Signal' in name:
            return name

    def read(self, path):
        with h5py.File(path + '.fast5', 'r') as h5_file:
            if 'Analyses/RawGenomeCorrected_000/BaseCalled_template/Events' not in h5_file:
                return []
            corrected_events = h5_file['Analyses/RawGenomeCorrected_000/BaseCalled_template/Events']
            corrected_events_array = corrected_events[()]
            raw_signal = h5_file['Raw/Reads']
            signal_path = raw_signal.visit(self.find_signal)
            dataset = raw_signal[signal_path][()]
            offset = corrected_events.attrs.get('read_start_rel_to_raw')
            event_position = np.array([x[2] + offset for x in corrected_events_array])
            event_length = np.array([x[3] for x in corrected_events_array])
            sequence = np.array([x[4] for x in corrected_events_array])
            i = self.parser.skip
            current_start = i
            current_len = 0
            examples = []
            while i < len(event_position) - self.parser.skip:
                if sequence[i] not in alphabet_dict.keys():
                    current_len = 0
                    current_start = i + 1
                    i += 1
                else:
                    if current_len + event_length[i] < self.parser.segment_length:
                        current_len += event_length[i]
                        i += 1
                    else:
                        if i - current_start > 4:
                            signal = dataset[event_position[current_start]: event_position[
                                                                                current_start] + self.parser.segment_length]
                            normalized_signal = (signal - np.mean(np.unique(dataset))) / np.std(np.unique(dataset))
                            breaks = event_position[current_start + 1:i - 1] - event_position[current_start]
                            current_seq = sequence[current_start:i - 1]
                            example = TrainingExample(self.parser.get_id(), path, current_start, normalized_signal,
                                                      signal, current_seq, breaks)
                            examples.append(example)
                        current_len = 0
                        current_start = i + 1
                        i += 1
            return examples


class ChironFileReader:

    def __init__(self, parser=None):
        self.parser = parser

    def read_for_eval(self, path):
        with open(path, 'r') as signal_file:
            dataset = signal_file.readlines()
            dataset = dataset[0].strip()
            dataset = dataset.split(' ')
            dataset = [int(x) for x in dataset]
            i = 0
            signal = []
            indices = []
            mean = np.mean(np.unique(dataset))
            std = np.std(np.unique(dataset))
            while i + 300 < len(dataset):
                raw = dataset[i:i + 300]
                normalized = (raw - mean) / std
                signal.append(normalized)
                indices.append(i)
                i = i + 30
            return signal, indices

    def filter_files(self, files):
        return [x for x in files if x.endswith('signal') or x.endswith('label')]

    def read(self, path):
        with open(path + '.label', 'r') as label_file:
            with open(path + '.signal', 'r') as signal_file:
                dataset = signal_file.readlines()
                dataset = dataset[0].strip()
                dataset = dataset.split(' ')
                dataset = [int(x) for x in dataset]
                corrected_events_array = label_file.readlines()
                corrected_events_array = [x.strip() for x in corrected_events_array]
                corrected_events_array = [x.split(' ') for x in corrected_events_array]
                event_position = np.array([int(x[0]) for x in corrected_events_array])
                event_length = np.array([int(x[1]) - int(x[0]) for x in corrected_events_array])
                sequence = np.array([x[2] for x in corrected_events_array])
                i = self.parser.skip
                current_start = i
                current_len = 0
                examples = []
                while i < len(event_position) - self.parser.skip:
                    if sequence[i] not in alphabet_dict.keys():
                        current_len = 0
                        current_start = i + 1
                        i += 1
                    else:
                        if current_len + event_length[i] < self.parser.segment_length:
                            current_len += event_length[i]
                            i += 1
                        else:
                            if i - current_start > 4:
                                signal = dataset[event_position[current_start]: event_position[
                                                                                    current_start] + self.parser.segment_length]
                                normalized_signal = (signal - np.mean(np.unique(dataset))) / np.std(np.unique(dataset))
                                breaks = event_position[current_start + 1:i - 1] - event_position[current_start]
                                current_seq = sequence[current_start:i - 1]
                                example = TrainingExample(self.parser.get_id(), path, current_start, normalized_signal,
                                                          signal, current_seq, breaks)
                                examples.append(example)
                            current_len = 0
                            current_start = i + 1
                            i += 1
                return examples


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

    def __init__(self, data_dir, file_list, batch_size=100):
        self.data_dir = data_dir
        self.files = os.listdir(data_dir)
        self.files = [x for x in self.files if x.endswith('.fast5') or x.endswith('.signal')]
        if file_list is not None:
            self.files = [x for x in self.files if x.split('.')[0] in file_list]
        self.output_types = (tf.float32, tf.int64, tf.string)
        self.output_shapes = (tf.TensorShape([None, 300, 1]), tf.TensorShape([None]), tf.TensorShape([None]))
        self.current_file = 0
        self.current_row = 0
        self.batch_size = batch_size
        self.current_file_data = None
        self.current_file_len = None

    def generator(self):
        while self.has_next_file():
            self.current_file_data = self.extract_next_file()
            self.current_row = 0
            self.current_file_len = self.current_file_data[1].shape[0]
            while self.current_row < self.current_file_len:
                self.current_row = self.current_row + 1
                yield (self.current_file_data[0][self.current_row - 1, :, :],
                       self.current_file_data[1][self.current_row - 1],
                       self.current_file_data[2][self.current_row - 1])

    def get_size(self):
        return len(self.files)

    def has_next_file(self):
        return self.current_file < len(self.files)

    def extract_next_file(self):
        filename = self.files[self.current_file]
        if filename.endswith('fast5'):
            reader = H5FileReader()
        elif filename.endswith('signal'):
            reader = ChironFileReader()
        else:
            raise ValueError(f"Format was {filename.split('.')[1]}, but it must be one of {', '.join(['.fast5', '.signal'])}.")
        signal, index = reader.read_for_eval(os.path.join(self.data_dir, filename))
        signal = np.expand_dims(np.stack(signal).astype(np.float32), -1)
        index = np.array(index)
        filenames = np.repeat(filename, index.shape[0])
        self.current_file = self.current_file + 1
        return signal, index, filenames

    def get_dataset(self):
        return tf.data.Dataset.from_generator(self.generator,
                                              self.output_types,
                                              self.output_shapes)


if __name__ == "__main__":
    parser = SignalFileParser(sys.argv[1], sys.argv[2], format=sys.argv[3])
    parser.create()
