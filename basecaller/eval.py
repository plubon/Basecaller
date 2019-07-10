#!/usr/bin/python3
from Network import get_default_model
from keras import optimizers
from keras.utils import multi_gpu_model
from keras.callbacks import CSVLogger
from keras.models import Model, load_model
import keras.backend as K
from DatasetManager import SignalSequence, DataDirectoryReader
import datetime
import os
import json
import numpy as np
from Levenshtein import distance, editops
from Data import Dataset, ExampleSequence, TrainingExample
import sys


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

def main(dataset_path, model_path):
    dataset = Dataset(dataset_path)
    with open(os.path.join(model_path, 'test.txt')) as test_file:
        test = [x.strip() for x in test_file.readlines()]
    test_seq = ExampleSequence(dataset, test, name='test', batch_size=16)
    model = load_model(os.path.join(model_path, 'model.h5'), custom_objects={'<lambda>': lambda y_true, y_pred: y_pred})
    model = multi_gpu_model(model, gpus=2)
    sub_model = model.get_layer('model_2')
    sub_model = sub_model.get_layer('model_1')
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
        val = K.ctc_decode(preds, np.full(16, batch['input_length'][0,0]), greedy=True)
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
    metrics_file_path = os.path.join(model_path, 'metrics1.json')
    write_dict_to_file(metrics_file_path, metrics)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])