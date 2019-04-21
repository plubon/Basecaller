#!/usr/bin/python3
from Network import get_default_model
from keras import optimizers
from keras.utils import multi_gpu_model
from keras.callbacks import CSVLogger
from keras.models import Model
import keras.backend as K
import datetime
import os
import json
import numpy as np
from Levenshtein import distance, editops
import sys
from Data import Dataset, ExampleSequence


def write_file_dict_to_file(path, file_dict):
    lines = []
    for k in file_dict.keys():
        lines += [os.path.join(k, x) for x in file_dict[k]]
    write_lines_to_file(path, lines)

def write_lines_to_file(path, lines):
    with open(path, 'w') as file:
        for line in lines:
            file.write("%s\n" % line)

def write_dict_to_file(path, params):
    with open(path, 'w') as file:
        json.dump(params, file)

def main(data_path, epochs):
    run_start_time = str(datetime.datetime.now())
    dataset = Dataset(data_path)
    train, test = dataset.train_test_split()
    signal_seq = ExampleSequence(dataset, train)
    test_seq = ExampleSequence(dataset, test)
    os.mkdir('runs/'+run_start_time)
    log_dir = os.path.join('runs', run_start_time)
    write_lines_file(os.path.join(log_dir,'test.txt'), test)
    write_lines_file(os.path.join(log_dir,'train.txt'), train)
    csv_logger = CSVLogger(os.path.join(log_dir, 'Log.csv'))
    model = get_default_model()
    model = multi_gpu_model(model, gpus=2)
    param = {'lr':0.001, 'beta_1':0.9, 'beta_2':0.999, 'epsilon':None, 'decay':0.001}
    param_file_path = os.path.join(log_dir, 'params.json')
    write_dict_to_file(param_file_path, param)
    adam = optimizers.Adam(**param)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred},optimizer=adam)
    model.fit_generator(signal_seq, validation_data=test_seq, epochs=epochs, callbacks=[csv_logger])
    model.save(os.path.join(log_dir, 'model.h5'))
    sub_model = model.get_layer('model_1')
    im_model = Model(inputs=sub_model.get_input_at(0), outputs =sub_model.get_layer('activation_1').output)
    dists = []
    ops = []
    lens = []
    pred_lens = []
    real = []
    predicted = []
    for j in range(len(test_seq)):
        batch = test_seq[j][0]
        preds = im_model.predict_on_batch(batch)
        val = K.ctc_decode(preds, np.full(100, batch['input_length'][0,0]), greedy=False)
        decoded = K.eval(val[0][0])
        for i in range(decoded.shape[0]):
            real_label = batch['the_labels'][i, :batch['label_length'][i,0]]
            real_label = ''.join([str(int(x)) for x in real_label.tolist()])
            pred_label = list(filter(lambda x: x!= -1, decoded[i,:].tolist()))
            pred_label = [str(x) for x in pred_label]
            pred_label = ''.join(pred_label)
            dists.append(distance(pred_label, real_label))
            ops.append(editops(pred_label, real_label))
            lens.append(len(real_label))
            pred_lens.append(len(pred_label))
            real.append(real_label)
            predicted.append(pred_label)
    op_counts = {'insert':0, 'replace':0, 'delete':0}
    for op in ops:
        for x in op:
            op_counts[x[0]] += 1
    for key in op_counts.keys():
        op_counts[key] = op_counts[key] / sum(lens)
    metrics = {
        'LER': sum(dists)/sum(lens),
        'real_mean_length': np.mean(lens),
        'predicted_mean_length': np.mean(pred_lens)
    }
    metrics.update(op_counts)
    metrics_file_path = os.path.join(log_dir, 'metrics.json')
    write_dict_to_file(metrics_file_path, metrics)

if __name__ == "__main__":
	main(sys.argv[1], int(sys.argv[2]))