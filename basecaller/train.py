#!/usr/bin/python3
from Network import get_model
from keras import optimizers
from basecaller.DatasetManager import SignalSequence, DataDirectoryReader

def main():
	dir_reader = DataDirectoryReader('Dane/squiggled/')
	train, test = dir_reader.get_train_test_files()
	signal_seq = SignalSequence(train)
	test_len = sum([len(test[x]) for x in test.keys()])
	test_seq = SignalSequence(test, number_of_reads=test_len)
	model = get_model()
	adam = optimizers.Adam()
	model.compile(loss={'ctc': lambda y_true, y_pred: y_pred},optimizer=adam)
	model.fit_generator(signal_seq, validation_data=test_seq)
	model.save('model.h5')


if __name__ == "__main__":
	main()
