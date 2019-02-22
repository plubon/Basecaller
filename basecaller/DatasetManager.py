import os
import random
import h5py
import numpy as np
import keras
from sklearn.model_selection import train_test_split

class SignalSequence(keras.utils.Sequence):

	def __init__(self, file_paths, batch_size=100, number_of_reads=4000, read_lens=[200,400,1000]):
		self.file_paths = file_paths
		self.read_lens = read_lens
		self.number_of_reads = number_of_reads
		self.used_files_paths = []
		self.batch_size = batch_size

	def __len__(self):
		return int(np.floor(self.number_of_reads/self.batch_size))

	def __getitem__(self, index):
		type_dirs = random.choices(self.file_paths.keys(), k=batch_size)
		read_len = random.choice(self.read_lens)

class DatasetReader:

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

class DatasetManager:

	def get_random_window_size(self):
		return random.choice([200, 400, 1000])

	def find_signal(self, name):
	    if 'Signal' in name:
	        return name

	def get_sample_from_h5(self, h5_file, window_size):
		corrected_events = h5_file['Analyses/RawGenomeCorrected_000/BaseCalled_template/Events']
		raw_signal = h5_file['Raw/Reads']
		signal_path = raw_signal.visit(self.find_signal)
		dataset = raw_signal[signal_path]
		offset = corrected_events.attrs.get('read_start_rel_to_raw')
		raw_signal = dataset[offset:]
		index = random.randrange(len(dataset) - window_size)
		signal = dataset[index:(index+window_size)]
		signal_index = corrected_events[0][2]
		found_first = False
		seq = []
		event_index = 0
		print(index, offset, len(dataset), window_size)
		print(signal_index)
		while signal_index < offset + index + window_size:
			if signal_index + corrected_events[event_index][3] > offset + index:
				found_first = True
				seq.append(corrected_events[event_index][4])
			signal_index = signal_index + corrected_events[event_index][3]
			event_index = event_index + 1
		return signal, seq


	def get_batch(self, path, batch_size):
		data_files = [file for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))]
		files = random.choices(data_files, k=batch_size)
		samples = []
		for file in files:
			with h5py.File(os.path.join(path, file),'r') as contents:	
				samples.append(self.get_sample_from_h5(contents, self.get_random_window_size()))
		return samples
