import os
import random
import h5py
import numpy as np
import keras
from math import ceil, floor
from sklearn.model_selection import train_test_split

class BaseSequence(keras.utils.Sequence):

	def __init__(self, file_paths, batch_size=100, number_of_reads=20000, read_lens=[200,400,1000], dir_probs=None):
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

	def __len__(self):
		return int(ceil(self.number_of_reads/self.batch_size))

class BoundarySequence(BaseSequence):

	def __getitem__(self, index):
		read_len =  self.read_lens[index % len(self.read_lens)]
		samples = [Fast5Reader(file_path, read_len).get_squiggle_boundary_sample() for file_path in self.batches[index]]
		return samples


class SignalSequence(BaseSequence):

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

	def seq_to_label(self, seq):
		labels = [self.alphabet_dict[x] for x in seq]
		return labels

	def __getitem__(self, index):
		read_len =  self.read_lens[index % len(self.read_lens)]
		samples = [Fast5Reader(file_path, read_len).get_sample() for file_path in self.batches[index]]
		X = [s[0] for s in samples]
		Y = [s[1] for s in samples]
		Y = [self.seq_to_label(y) for y in Y]
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

	index_filename = 'index'

	def __init__(self, path):
		self.base_path = path
		self.type_dirs = os.listdir(self.base_path)
		self.files = dict()
		self.stuck_files = []
		self.uncertain_files = []
		for type_dir in self.type_dirs:
			dir_path = os.path.join(self.base_path, type_dir)
			if os.path.isfile(os.path.join(dir_path, self.index_filename)):
				with open(os.path.join(dir_path, self.index_filename)) as file:
					valid_files_names = file.readlines()
					self.files[type_dir] = [x.strip() for x in valid_files_names]
			else:
				self.files[type_dir] = [file for file in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path,file)) and self.file_is_valid(type_dir, file)]
				with open(os.path.join(dir_path, self.index_filename), 'w') as file:
					for filename in self.files[type_dir]:
						file.write("%s\n" % filename)
		to_remove = []
		for type_dir in self.type_dirs:
			if len(self.files[type_dir]) == 0:
				to_remove.append(type_dir)
		for val in to_remove:
			self.type_dirs.remove(val)
			self.files.pop(val)

	def get_stuck_files(self):
		for type_dir in self.type_dirs:
			dir_path = os.path.join(self.base_path, type_dir)
			self.stuck_files += [file for file in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path,file)) and self.file_is_stuck(type_dir, file)]
		return self.stuck_files

	def get_uncertain_files(self):
		for type_dir in self.type_dirs:
			dir_path = os.path.join(self.base_path, type_dir)
			self.uncertain_files += [file for file in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path,file)) and self.file_is_uncertain(type_dir, file)]
		return self.uncertain_files		

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
				lens = [x[3] for x in rows]				
				return max(lens) > 200
			else:
				return False

	def file_is_uncertain(self, dir_name, file):
		with h5py.File(os.path.join(self.base_path, dir_name, file),'r') as contents:
			has_analysis = 'Analyses/RawGenomeCorrected_000/BaseCalled_template/Events' in contents
			if has_analysis:
				rows = contents['Analyses/RawGenomeCorrected_000/BaseCalled_template/Events']
				chars = [x[4] for x in rows]				
				return any([x not in b'AaCcGgTt' for x in chars])
			else:
				return False

	def file_has_correct_length(self, dir_name, file):
		with h5py.File(os.path.join(self.base_path, dir_name, file),'r') as contents:
			has_analysis = 'Analyses/RawGenomeCorrected_000/BaseCalled_template/Events' in contents
			if has_analysis:
				return contents['Analyses/RawGenomeCorrected_000/BaseCalled_template/Events'][-1][2] > 1000
			else:
				return False

	def file_is_valid(self, dir_name, file):
		with h5py.File(os.path.join(self.base_path, dir_name, file),'r') as contents:
			has_analysis = 'Analyses/RawGenomeCorrected_000/BaseCalled_template/Events' in contents
			if has_analysis:
				data = contents['Analyses/RawGenomeCorrected_000/BaseCalled_template/Events']
				lens = [x[3] for x in data]
				chars = [x[4] for x in data]
				return all([x in b'AaCcGgTt' for x in chars]) and max(lens) < 200 and data[-1][2] > 1000
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

	def get_squiggle_boundary_sample(self):
		with h5py.File(self.path,'r') as contents:
			return self._get_squiggle_boundary_sample_from_h5(contents)
	
	def _get_squiggle_boundary_sample_from_h5(self, h5_file):
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
		seq_end = max(np.argmax(event_position > index + self.window_size), seq_start+1)
		seq = sequence[seq_start:seq_end]
		boundaries = np.zeros(self.window_size)
		boundaries_indices = event_position - index
		boundaries_indices = np.extract(np.logical_and(boundaries_indices > 0, boundaries_indices < self.window_size), boundaries_indices)
		boundaries[boundaries_indices] = 1
		return signal, seq, boundaries

	def _get_boundary_sample_from_h5(self, h5_file):
		corrected_events = h5_file['Analyses/RawGenomeCorrected_000/BaseCalled_template/Events']
		corrected_events_array = corrected_events[()]
		event_position = np.array([x[2] for x in corrected_events_array])
		raw_signal = h5_file['Raw/Reads']
		signal_path = raw_signal.visit(self.find_signal)
		dataset = raw_signal[signal_path]
		offset = corrected_events.attrs.get('read_start_rel_to_raw')
		event_position = np.array([x[2] for x in corrected_events_array])
		analyzed_length = event_position[-1]
		index = random.randrange(analyzed_length - self.window_size)
		signal = dataset[(offset+index):(offset+index+self.window_size)]
		boundaries = np.zeros(self.window_size)
		boundaries_indices = event_position - index
		boundaries_indices = np.extract(np.logical_and(boundaries_indices > 0, boundaries_indices < self.window_size), boundaries_indices)
		boundaries[boundaries_indices] = 1
		return signal, boundaries


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
		seq_end = max(np.argmax(event_position > index + self.window_size), seq_start+1)
		seq = sequence[seq_start:seq_end]
		return signal, seq