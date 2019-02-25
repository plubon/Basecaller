import os
import random
import h5py
import numpy as np
import keras
from math import ceil
from sklearn.model_selection import train_test_split

class SignalSequence(keras.utils.Sequence):

	alphabet = 'ACGT '

	def __init__(self, file_paths, batch_size=100, number_of_reads=4000, read_lens=[200,400,1000], dir_probs=None):
		self.file_paths = file_paths
		self.read_lens = read_lens
		self.number_of_reads = number_of_reads
		self.batch_size = batch_size
		self.batch_read_lens = random.choices(read_lens, k=len(self))
		if dir_probs is None:
			self.dir_probs = [1/len(self.file_paths.keys())]*len(self.file_paths.keys())
		else:
			self.dir_probs = dir_probs
		self.dir_counts = {key:floor(batch_size*value) for (key, value) in self.dir_probs}
		if sum(self.dir_counts.values()) < self.batch_size:
			add_to_keys = random.choices(self.dir_counts.keys(), weights=self.dir_probs, k=sum(self.dir_counts.values())-self.batch_size)
			for key in add_to_keys:
				self.dir_counts[key] += 1
		self.dir_sequences = dict()
		for key in self.dir_counts.keys():
			no_of_reps = ceil((self.dir_counts[key] * len(self)) / len(self.file_paths[key]))
			self.dir_sequences[key] = []
			for _ in range(no_of_reps):
				random.shuffle(self.file_paths[key])
				self.dir_sequences[key] += self.file_paths[key]
		self.batches = []
		for i in range(len(self)):
			batch = []
			for key in self.file_paths.keys():
				batch += self.dir_sequences[key][self.dir_counts*i:self.dir_counts*(i+1)]
			random.shuffle(batch)
			self.batches.append(batch)

	def seq_to_label(self, seq):
		return [self.alphabet.find(x) for x in seq]

	def __len__(self):
		return int(ceil(self.number_of_reads/self.batch_size))

	def __getitem__(self, index):
		read_len =  self.read_lens[index % len(self.read_lens)]
		samples = [Fast5Reader(file_path, read_len).get_sample() for file_path in self.batches[index]]
		X = [s[0] for s in samples]
		Y = [s[1] for s in samples]
		Y = [self.seq_to_label(y) for y in Y]
		max_len = max([len(y) for y in Y])
		Y_array = np.full((len(Y), max_len), -1)
		for idx, word in enumerate(len(Y)):
			Y_array[idx, 0:len(word)] = word 
		inputs = {
			'the_input': np.stack(X),
			'the_labels': Y_array,
			'input_length': read_len,
			'label_length': max_len
		}
		outputs = {'ctc': np.zeros([self.batch_size])}
		return (inputs, outputs)

class DataDirectoryReader:

	def __init__(self, path):
		self.base_path = path
		self.type_dirs = os.listdir(self.path)
		self.files = dict()
		for type_dir in self.type_dirs:
			dir_path = os.path.join(self.base_path, type_dir)
			self.files[type_dir] = [file for file in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path,file))]

	def get_train_test_files(self):
		train = dict()
		test = dict()
		for type_dir in self.type_dirs:
			train, test = train_test_split(self.files[type_dir], test_size=0.2)
			train[type_dir] = [os.path.join(self.base_path, type_dir, x) for x in train]
			test[type_dir] = [os.path.join(self.base_path, type_dir, x) for x in test]
		return train, test

class Fast5Reader:

	def __init__(self, path, window_size):
		self.path = path
		self.window_size = window_size

	def find_signal(self, name):
	    if 'Signal' in name:
	        return name

    def get_sample(self):
    	with h5py.File(os.path.join(path, file),'r') as contents:
    		return self._get_sample_from_h5(contents)

	def _get_sample_from_h5(self, h5_file):
		corrected_events = h5_file['Analyses/RawGenomeCorrected_000/BaseCalled_template/Events']
		raw_signal = h5_file['Raw/Reads']
		signal_path = raw_signal.visit(self.find_signal)
		dataset = raw_signal[signal_path]
		offset = corrected_events.attrs.get('read_start_rel_to_raw')
		raw_signal = dataset[offset:]
		index = random.randrange(len(dataset) - self.window_size)
		signal = dataset[index:(index+self.window_size)]
		signal_index = corrected_events[0][2]
		found_first = False
		seq = []
		event_index = 0
		while signal_index < offset + index + self.window_size:
			if signal_index + corrected_events[event_index][3] > offset + index:
				found_first = True
				seq.append(corrected_events[event_index][4])
			signal_index = signal_index + corrected_events[event_index][3]
			event_index = event_index + 1
		return signal, seq