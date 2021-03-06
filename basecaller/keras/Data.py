import os
import h5py
import numpy as np
import shelve
import sys
import datetime
import keras
from math import ceil, floor
from sklearn.model_selection import train_test_split
import random

class ExampleSequence(keras.utils.Sequence):

    alphabet_dict = {
	b'A':0,
	b'a':0,
	b'C':1,
	b'c':1,
	b'G':2,
	b'g':2,
	b'T':3,
	b't':3,
    'A':0,
	'a':0,
	'C':1,
	'c':1,
	'G':2,
	'g':2,
	'T':3,
	't':3
	}

    def __init__(self, dataset, ids, name='', batch_size=150):
        self.name = name
        self.dataset = dataset
        self.ids = ids
        self.batch_size = batch_size
        random.shuffle(self.ids)

    def seq_to_label(self, seq):
        labels = [self.alphabet_dict[x] for x in seq]
        return labels

    def __len__(self):
        return int(floor(len(self.ids)/self.batch_size))

    def __getitem__(self, index):
        examples = self.dataset.get(self.ids[self.batch_size*index:self.batch_size*(index+1)])
        X = [x.signal for x in examples]
        Y = [self.seq_to_label(x.sequence) for x in examples]
        input_lengths = [len(x) for x in X]
        label_lengths = [len(x) for x in Y]
        Y_array = np.zeros((len(Y), 300))
        for idx, word in enumerate(Y):
            Y_array[idx, 0:len(word)] = word
        input_arr = np.expand_dims(np.stack(X).astype(np.float32), -1)
        inputs = {
			'the_input': input_arr,
			'the_labels': Y_array,
			'input_length': np.reshape(np.array(input_lengths), [input_arr.shape[0], 1]),
			'label_length': np.reshape(np.array(label_lengths), [input_arr.shape[0], 1])
		}
        outputs = {'ctc': np.zeros([len(examples)])}
        return (inputs, outputs)

    def on_epoch_end(self):
        random.shuffle(self.ids)

class EvalDataset:

    alphabet_dict = {
	b'A':0,
	b'a':0,
	b'C':1,
	b'c':1,
	b'G':2,
	b'g':2,
	b'T':3,
	b't':3,
    'A':0,
	'a':0,
	'C':1,
	'c':1,
	'G':2,
	'g':2,
	'T':3,
	't':3
	}

    def __init__(self, path, seq_len=300, skip=10):
        self.skip = skip
        self.path = path
        self.seq_len = seq_len
        self.files = os.listdir(self.path)
        self.files = [x.split('.')[0] for x in self.files if x != '']
        self.files = list(set(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with open(os.path.join(self.path, self.files[index])+'.label', 'r') as label_file:
            with open(os.path.join(self.path,self.files[index])+'.signal', 'r') as signal_file:
                dataset = signal_file.readlines()
                dataset = dataset[0].strip()
                dataset = dataset.split(' ')
                dataset = [int(x) for x in dataset]
                corrected_events_array = label_file.readlines()
                corrected_events_array = [x.strip() for x in corrected_events_array]
                corrected_events_array = [x.split(' ') for x in corrected_events_array]
                event_start = [int(x[0]) for x in corrected_events_array]
                event_end = [int(x[1])  for x in corrected_events_array]
                sequence = np.array([x[2] for x in corrected_events_array])
                signal = dataset[event_start[self.skip]:event_end[-self.skip]]
                signal = (signal - np.mean(dataset))/np.std(dataset)
                bases = sequence[self.skip:-self.skip]
                return signal, bases

class Dataset:
    
    def __init__(self, path):
        self.path = path
        with shelve.open(self.path, 'r') as db:
            self.size = db['size']

    def train_test_split(self, test_size=0.2):
        ids = np.arange(1, self.size + 1)
        train, test = train_test_split(ids, test_size=test_size)
        return train, test

    def get(self, ids):
        examples = []
        with shelve.open(self.path) as db:
            for id in ids:
                examples.append(db[str(id)])
        return examples

class TrainingExample:

    def __init__(self, id, filename, index, signal, raw_signal, sequence, breaks):
        self.id = id
        self.filename = filename
        self.index = index
        self.signal = signal
        self.sequence = sequence
        self.breaks = breaks
        self.raw_signal = raw_signal

class ChironDatasetCreator:

    alphabet_dict = {
	b'A':0,
	b'a':0,
	b'C':1,
	b'c':1,
	b'G':2,
	b'g':2,
	b'T':3,
	b't':3,
    'A':0,
	'a':0,
	'C':1,
	'c':1,
	'G':2,
	'g':2,
	'T':3,
	't':3
	}

    def __init__(self, input_path, output_path, seg_length=300, skip=10):
        self.input_path = input_path
        self.output_path = output_path
        self.files = os.listdir(self.input_path)
        self.files = [x.split('.')[0] for x in self.files]
        self.files = list(set(self.files))
        print(len(self.files))
        self.skip = skip
        self.segment_length = seg_length
        self.id_counter = 1
        
    def create(self):
        sample_count = 0
        with shelve.open(self.output_path) as db:
            for idx, input_file in enumerate(self.files):
                if idx % 100 == 0:
                    print(idx)
                samples = self.get_samples_from_file(os.path.join(self.input_path, input_file))
                sample_count += len(samples)
                for sample in samples:
                    db[sample.id] = sample
            db['size'] = sample_count
        print('Created '+str(sample_count)+' samples.')
    
    def get_id(self):
        id = self.id_counter
        self.id_counter += 1
        return str(id)

    def find_signal(self, name):
	    if 'Signal' in name:
	        return name

    def get_samples_from_file(self, path):
        with open(path+'.label', 'r') as label_file:
            with open(path+'.signal', 'r') as signal_file:
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
                i = self.skip
                current_start = i
                current_len = 0
                examples = []
                while i < len(event_position) - self.skip:
                    if sequence[i] not in self.alphabet_dict.keys():
                        current_len = 0
                        current_start = i + 1
                        i += 1
                    else:
                        if current_len + event_length[i] < self.segment_length:
                            current_len += event_length[i]
                            i += 1 
                        else:
                            if i - current_start >4 :
                                signal = dataset[event_position[current_start]: event_position[current_start]+self.segment_length]
                                normalized_signal = (signal - np.mean(np.unique(dataset)))/np.std(np.unique(dataset))
                                breaks = event_position[current_start+1:i-1] - event_position[current_start]
                                current_seq = sequence[current_start:i-1]
                                example = TrainingExample(self.get_id(), path, current_start, normalized_signal, signal, current_seq, breaks)
                                examples.append(example)
                            current_len = 0
                            current_start = i + 1
                            i += 1
                return examples

class DatasetCreator:

    alphabet_dict = {
	b'A':0,
	b'a':0,
	b'C':1,
	b'c':1,
	b'G':2,
	b'g':2,
	b'T':3,
	b't':3
	}

    def __init__(self, input_path, output_path, seg_length=300, skip=10):
        self.input_path = input_path
        self.output_path = output_path
        self.files = os.listdir(self.input_path)
        self.skip = skip
        self.segment_length = seg_length
        self.id_counter = 1
        
    def create(self):
        sample_count = 0
        with shelve.open(self.output_path) as db:
            for input_file in self.files:
                if input_file.endswith('fast5'):
                    samples = self.get_samples_from_file(os.path.join(self.input_path, input_file))
                    sample_count += len(samples)
                    for sample in samples:
                        db[sample.id] = sample
            db['size'] = sample_count
        print('Created '+str(sample_count)+' samples.')
    
    def get_id(self):
        id = self.id_counter
        self.id_counter += 1
        return str(id)

    def find_signal(self, name):
	    if 'Signal' in name:
	        return name

    def get_samples_from_file(self, path):
        with h5py.File(path, 'r') as h5_file:
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
            i = self.skip
            current_start = i
            current_len = 0
            examples = []
            while i < len(event_position) - self.skip:
                if sequence[i] not in self.alphabet_dict.keys():
                    current_len = 0
                    current_start = i + 1
                    i += 1
                else:
                    if current_len + event_length[i] < self.segment_length:
                        current_len += event_length[i]
                        i += 1 
                    else:
                        if i - current_start >4 :
                            signal = dataset[event_position[current_start]: event_position[current_start]+self.segment_length]
                            normalized_signal = (signal - np.mean(np.unique(dataset)))/np.std(np.unique(dataset))
                            breaks = event_position[current_start+1:i-1] - event_position[current_start]
                            current_seq = sequence[current_start:i-1]
                            example = TrainingExample(self.get_id(), path, current_start, normalized_signal, signal, current_seq, breaks)
                            examples.append(example)
                        current_len = 0
                        current_start = i + 1
                        i += 1
            return examples

if __name__ == "__main__":
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    creator = ChironDatasetCreator(sys.argv[1], sys.argv[2])
    creator.create()
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    f = open("end.txt","w+")
    f.write(now.strftime("%Y-%m-%d %H:%M:%S"))
    f.close()

