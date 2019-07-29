#!/usr/bin/python3
from Network import get_default_model
from keras import optimizers
from keras.utils import multi_gpu_model
from keras.callbacks import CSVLogger
from DatasetManager import SignalSequence, DataDirectoryReader
from itertools import product

def main():
	lrs = [0.001, 0.01, 0.0001]
	epsilon = [None, 0.1, 1]
	b1 = [0.8, 0.9, 0.99]
	b2 = [0.9, 0.99, 0.999]
	dir_reader = DataDirectoryReader('../Dane/squiggled/')
	train, test = dir_reader.get_train_test_files()
	params = product(lrs, b1, b2, epsilon)
	for param in params:
		csv_logger = CSVLogger('Log/'+str(param)+'.csv')
		signal_seq = SignalSequence(train, batch_size=150)
		test_len = sum([len(test[x]) for x in test.keys()])
		test_seq = SignalSequence(test, number_of_reads=test_len)
		model = get_default_model()
		model = multi_gpu_model(model, gpus=2)
		adam = optimizers.Adam(lr=param[0], beta_1=param[1], beta_2=param[2], epsilon=param[3])
		model.compile(loss={'ctc': lambda y_true, y_pred: y_pred},optimizer=adam)
		model.fit_generator(signal_seq, validation_data=test_seq, epochs=10, callbacks=[csv_logger])


if __name__ == "__main__":
	main()
