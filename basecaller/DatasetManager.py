import os
import random
import h5py
import numpy as np
import keras
from math import ceil, floor
from sklearn.model_selection import train_test_split

class SignalSequence(keras.utils.Sequence):

	alphabet = b'ACGT '

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
		self.dir_counts = {list(file_paths.keys())[idx]:floor(batch_size*value) for idx, value in enumerate(self.dir_probs)}
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
				batch += self.dir_sequences[key][self.dir_counts[key]*i:self.dir_counts[key]*(i+1)]
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
		Y_array = np.zeros((len(Y), 300))
		for idx, word in enumerate(Y):
			Y_array[idx, 0:len(word)] = word
		input_arr = np.expand_dims(np.stack(X).astype(np.float32), -1)
		for i in range(input_arr.shape[0]):
			input_arr[i,:] = (input_arr[i,:] - np.mean(input_arr[i,:]))/np.std(input_arr[i,:])
		inputs = {
			'the_input': input_arr,
			'the_labels': Y_array,
			'input_length': np.full([input_arr.shape[0], 1], read_len),
			'label_length': np.reshape(np.array([len(y) for y in Y]), [input_arr.shape[0], 1])
		}
		outputs = {'ctc': np.zeros([self.batch_size])}
		return (inputs, outputs)

class DataDirectoryReader:

	def __init__(self, path):
		self.base_path = path
		self.type_dirs = os.listdir(self.base_path)
		self.files = dict()
		for type_dir in self.type_dirs:
			dir_path = os.path.join(self.base_path, type_dir)
			self.files[type_dir] = [file for file in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path,file)) and self.file_is_valid(type_dir, file)]

	def get_stuck_files(self):
		for type_dir in self.type_dirs:
			dir_path = os.path.join(self.base_path, type_dir)
			self.files[type_dir] = [file for file in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path,file)) and self.file_is_stuck(type_dir, file)]

	def get_train_test_files(self):
		train_dict = dict()
		test_dict = dict()
		for type_dir in self.type_dirs:
			train, test = train_test_split(self.files[type_dir], test_size=0.2)
			train_dict[type_dir] = [os.path.join(self.base_path, type_dir, x) for x in train]
			test_dict[type_dir] = [os.path.join(self.base_path, type_dir, x) for x in test]
		return train_dict, test_dict
	
	def file_is_stuck(self, dir_name, file):
		with h5py.File(os.path.join(self.base_path, dir_name, file),'r') as contents:
			has_analysis = 'Analyses/RawGenomeCorrected_000/BaseCalled_template/Events' in contents
			if has_analysis:
				rows = contents['Analyses/RawGenomeCorrected_000/BaseCalled_template/Events']
				lens = [x[2] for x in rows]				
				return max(lens) > 50
			else:
				return False

	def file_is_valid(self, dir_name, file):
		with h5py.File(os.path.join(self.base_path, dir_name, file),'r') as contents:
			has_analysis = 'Analyses/RawGenomeCorrected_000/BaseCalled_template/Events' in contents
			if has_analysis:
				return contents['Analyses/RawGenomeCorrected_000/BaseCalled_template/Events'][-1][2] > 1000
			else:
				return False

class Fast5Reader:

	def __init__(self, path, window_size):
		self.path = path
		self.window_size = window_size

	def find_signal(self, name):
	    if 'Signal' in name:
	        return name

	def get_sample(self):
		with h5py.File(self.path,'r') as contents:
			return self._get_sample_from_h5(contents)

	def _get_sample_from_h5(self, h5_file):
		corrected_events = h5_file['Analyses/RawGenomeCorrected_000/BaseCalled_template/Events']
		corrected_events_array = corrected_events[()]
		raw_signal = h5_file['Raw/Reads']
		signal_path = raw_signal.visit(self.find_signal)
		dataset = raw_signal[signal_path]
		offset = corrected_events.attrs.get('read_start_rel_to_raw')
		event_position = np.array([x[2] for x in corrected_events_array])
		sequence = np.array([x[4] for x in corrected_events_array])
		analyzed_length = event_position[-1]
		index = random.randrange(analyzed_length - self.window_size)
		signal = dataset[(offset+index):(offset+index+self.window_size)]
		seq_start = np.argmax(event_position > index)
		seq_end = np.argmax(event_position > index + self.window_size)
		seq = sequence[seq_start:seq_end]
		if len(seq) == 0:
			print(self.path)
			print(analyzed_length)
			print(index)
			print(self.window_size)
			raise NameError('Sequence of length 0 created')
		return signal, seq