import os
import random
import h5py
import numpy as np
import keras

class SignalSequence(keras.utils.Sequence):

	def __init__(self, path, read_number=4000, read_lens=[200,400,100]):
		self.path = path
		self.read_number = read_number
		self.read_lens = read_lens
		self.used_files_paths = []

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
