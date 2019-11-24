import numpy as np
import h5py
from utils import alphabet_dict
from entities import TrainingExample
import random

class H5FileReader:

    def __init__(self, parser=None):
        self.parser = parser

    def filter_files(self, files):
        return [x for x in files if x.endswith('fast5')]

    def find_signal(self, name):
        if 'Signal' in name:
            return name

    def read_entire_sequence(self, path):
        with h5py.File(path, 'r') as h5_file:
            if 'Analyses/RawGenomeCorrected_000/BaseCalled_template/Events' not in h5_file:
                return []
            corrected_events = h5_file['Analyses/RawGenomeCorrected_000/BaseCalled_template/Events']
            corrected_events_array = corrected_events[()]
            sequence = np.array([x[4] for x in corrected_events_array])
            for idx, char in enumerate(sequence):
                if char not in alphabet_dict.keys():
                    sequence[idx] = b'A'
            return ''.join([x.decode("utf-8") for x in sequence])

    def read_for_eval(self, path):
        with h5py.File(path + '.fast5', 'r') as h5_file:
            if 'Analyses/RawGenomeCorrected_000/BaseCalled_template/Events' not in h5_file:
                return [], [], []
            corrected_events = h5_file['Analyses/RawGenomeCorrected_000/BaseCalled_template/Events']
            corrected_events_array = corrected_events[()]
            raw_signal = h5_file['Raw/Reads']
            signal_path = raw_signal.visit(self.find_signal)
            dataset = raw_signal[signal_path][()]
            offset = corrected_events.attrs.get('read_start_rel_to_raw')
            event_position = np.array([x[2] + offset for x in corrected_events_array])
            event_length = np.array([x[3] for x in corrected_events_array])
            sequence = np.array([x[4] for x in corrected_events_array])
            segments = []
            labels = []
            indices = []
            signal_index = event_position[0]
            label_index = 0
            end = event_position[-1] + event_length[-1]
            while signal_index + 300 < end and signal_index+300 < len(dataset):
                signal = dataset[signal_index:signal_index+300]
                normalized_signal = (signal - np.mean(np.unique(dataset))) / np.std(np.unique(dataset))
                label_end = label_index + 1
                while label_end < len(event_position) and event_position[label_end] <= signal_index + 300:
                    label_end = label_end + 1
                label = sequence[label_index:label_end]
                for idx, char in enumerate(label):
                    if char not in alphabet_dict.keys():
                        label[idx] = b'A'
                segments.append(normalized_signal)
                labels.append([x.decode("utf-8") for x in label])
                indices.append(signal_index)
                signal_index = signal_index + 30
                while event_position[label_index + 1] < signal_index:
                    label_index = label_index + 1
            return segments, labels, indices

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

    def filter_files(self, files):
        return [x for x in files if x.endswith('signal') or x.endswith('label')]

    def read_entire_sequence(self, path):
        with open(path, 'r') as label_file:
            corrected_events_array = label_file.readlines()
            corrected_events_array = [x.strip() for x in corrected_events_array]
            corrected_events_array = [x.split(' ') for x in corrected_events_array]
            sequence = np.array([x[2] for x in corrected_events_array])
            for idx, char in enumerate(sequence):
                if char not in alphabet_dict.keys():
                    sequence[idx] = 'A'
            return ''.join(sequence)

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

    def read_for_eval(self, path):
        with open(path + '.signal', 'r') as signal_file:
            dataset = signal_file.readlines()
            dataset = dataset[0].strip()
            dataset = dataset.split(' ')
            dataset = [int(x) for x in dataset]
            segments = []
            indices = []
            signal_index = 40
            while signal_index + 300 < len(dataset) - 40:
                signal = dataset[signal_index:signal_index + 300]
                normalized_signal = (signal - np.mean(np.unique(dataset))) / np.std(np.unique(dataset))
                segments.append(normalized_signal)
                indices.append(signal_index)
                signal_index = signal_index + 30
            return segments, indices


class DirtyChironFileReader:

    def __init__(self, parser=None):
        self.parser = parser

    def filter_files(self, files):
        return [x for x in files if x.endswith('signal') or x.endswith('label')]

    def read(self, path):
        skip = 4
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
                i = skip
                current_start = i
                current_len = 0
                examples = []
                while i < len(event_position) - skip:
                    if sequence[i] not in alphabet_dict.keys():
                        current_len = 0
                        current_start = i + 1
                        i += 1
                    else:
                        if current_len + event_length[i] <= self.parser.segment_length:
                            current_len += event_length[i]
                            i += 1
                        else:
                            if i - current_start > 2:
                                max_back = min(event_length[current_start-1], current_len + event_length[i] - self.parser.segment_length)
                                back = random.randint(0, max_back)
                                signal = dataset[event_position[current_start]-back: event_position[
                                                                                    current_start] + self.parser.segment_length - back]
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


class OverlapChironFileReader:

    def __init__(self, parser=None):
        self.parser = parser

    def filter_files(self, files):
        return [x for x in files if x.endswith('signal') or x.endswith('label')]

    def read(self, path):
        skip = 4
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
                sequence = np.array([x[2] for x in corrected_events_array])
                i = skip
                current_signal_start = event_position[i]
                j = i
                while event_position[j] <= current_signal_start + 300:
                    j = j + 1
                signal_end = event_position[-skip]
                examples = []
                while current_signal_start + 300 < signal_end:
                    current_seq = sequence[i:j - 1]
                    if all([x in alphabet_dict.keys() for x in current_seq]) and len(current_seq)>2:
                        signal = dataset[current_signal_start:current_signal_start+300]
                        normalized_signal = (signal - np.mean(np.unique(dataset))) / np.std(np.unique(dataset))
                        breaks = event_position[i + 1:j - 1] - event_position[i]
                        example = TrainingExample(self.parser.get_id(), path, i, normalized_signal,
                                                  signal, current_seq, breaks)
                        examples.append(example)
                    current_signal_start = current_signal_start + 10
                    while event_position[i] < current_signal_start:
                        i = i + 1
                    while event_position[j] < current_signal_start + 300:
                        j = j + 1
                return examples
