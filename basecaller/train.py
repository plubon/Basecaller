#!/usr/bin/python3
from Network import get_model
from keras import optimizers
from keras.utils import multi_gpu_model
from keras.callbacks import CSVLogger
from DatasetManager import SignalSequence, DataDirectoryReader
import datetime
import os
import json
import numpy as np


def write_file_dict_to_file(path, file_dict):
    lines = []
    for k in file_dict.keys():
        lines += [os.path.join(k, x) for x in file_dict[k]]
    write_lines_to_file(path, lines)

def write_lines_to_file(path, lines):
    with open(path, 'w') as file:
        for line in lines:
            file.write("%s\n" % line)

def write_params_to_file(path, params):
    with open(path, 'w') as file:
        json.dump(params, file)

def main():
    run_start_time = str(datetime.datetime.now())
    os.mkdir('runs/'+run_start_time)
    log_dir = os.path.join('runs', run_start_time)
    dir_reader = DataDirectoryReader('../Dane/squiggled/')
    train, test = dir_reader.get_train_test_files()
    write_file_dict_to_file(os.path.join(log_dir,'test.txt'), test)
    write_file_dict_to_file(os.path.join(log_dir,'train.txt'), train)
    csv_logger = CSVLogger(os.path.join(log_dir, 'Log.csv'))
    signal_seq = SignalSequence(train, batch_size=150)
    test_len = sum([len(test[x]) for x in test.keys()])
    test_seq = SignalSequence(test, number_of_reads=test_len)
    model = get_model()
    model = multi_gpu_model(model, gpus=2)
    param = {'lr':0.001, 'beta_1':0.9, 'beta_2':0.999, 'epsilon':None, 'decay':0.001}
    param_file_path = os.path.join(log_dir, 'params.json')
    write_params_to_file(param_file_path, param)
    adam = optimizers.Adam(**param)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred, 'activation_1':lambda y_true, y_pred: y_pred},optimizer=adam, loss_weights=[1,0])
    model.fit_generator(signal_seq, validation_data=test_seq, epochs=10, callbacks=[csv_logger])
    model.save(os.path.join(log_dir, 'model.h5'))

if __name__ == "__main__":
	main()