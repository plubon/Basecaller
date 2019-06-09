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
from Data import Dataset, ExampleSequence, TrainingExample, EvalDataset
import sys
import difflib


alphabet_dict = {
	b'A':0,
	b'a':0,
	b'C':1,
	b'c':1,
	b'G':2,
	b'g':2,
	b'T':3,
	b't':3,
    'A':0,
	'a':0,
	'C':1,
	'c':1,
	'G':2,
	'g':2,
	'T':3,
	't':3
	}

batch_size = 150
segment_length = 300
window_offset = segment_length // 10
def idx_to_label(index_list):
    bases = ['A', 'C', 'G', 'T']
    return [bases[x] for x in index_list]

def simple_assembly(bpreads):
    concensus = np.zeros([4,1000])
    pos = 0
    length = 0
    census_len = 1000
    for indx,bpread in enumerate(bpreads):
        if indx==0:
            add_count(concensus,0,bpread)
            continue
        d = difflib.SequenceMatcher(None,bpreads[indx-1],bpread)
        match_block = max(d.get_matching_blocks(),key = lambda x:x[2])
        disp = match_block[0]-match_block[1]
        if disp+pos+len(bpreads[indx])>census_len:
            concensus = np.lib.pad(concensus,((0,0),(0,1000)),mode = 'constant',constant_values = 0)
            census_len +=1000
        add_count(concensus,pos+disp,bpreads[indx])
        pos+=disp
        length = max(length,pos+len(bpreads[indx]))
    return concensus[:,:length]
    
def add_count(concensus,start_indx,segment):
    base_dict = {'A':0,'C':1,'G':2,'T':3,'a':0,'c':1,'g':2,'t':3}
    if start_indx<0:
        segment = segment[-start_indx:]
        start_indx = 0
    for i,base in enumerate(segment):
        concensus[base_dict[base]][start_indx+i] += 1

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
    dataset = EvalDataset(dataset_path)
    model = load_model(os.path.join(model_path, 'model.h5'), custom_objects={'<lambda>': lambda y_true, y_pred: y_pred})
    model = multi_gpu_model(model, gpus=2)
    sub_model = model.get_layer('model_2')
    sub_model = sub_model.get_layer('model_1')
    im_model = Model(inputs=sub_model.get_input_at(0), outputs =sub_model.get_layer('activation_1').output)
    dists = []
    lens = []
    pred_lens = []
    real = []
    predicted = []
    for f in range(len(dataset)):
        signal, real_label = dataset[f]
        reads = []
        for idx in range(0, len(signal)-(len(signal)%(batch_size*window_offset))-segment_length, batch_size*window_offset):
            batch_signal = [signal[idx+i:idx+i+batch_size] for i in range(0, batch_size*window_offset, window_offset)]
            for idx, s in enumerate(batch_signal):
                batch_signal[idx] = np.pad(s, (0, segment_length-len(s)), mode='edge')
            batch_signal_arr = np.expand_dims(np.stack(batch_signal).astype(np.float32), -1)
            batch = {
                'the_input':batch_signal_arr,
                'input_length':np.full((batch_size,1), segment_length),
                'label_length':np.full((batch_size,1), segment_length),
                'the_labels':np.zeros((150, 300))
            }
            preds = im_model.predict_on_batch(batch)
            val = K.ctc_decode(preds, np.full(150, segment_length), greedy=False)
            decoded = K.eval(val[0][0])
            bases = [''.join(idx_to_label(d)) for d in decoded]
            reads += bases
        assembly = simple_assembly(reads)
        c_bpread = idx_to_label(np.argmax(assembly,axis = 0))
        real_label = ''.join(real_label) 
        pred_label = ''.join(c_bpread)
        real.append(real_label)
        predicted.append(pred_label)
        dists.append(distance(pred_label, real_label))
        lens.append(len(real_label))
        pred_lens.append(len(pred_label))
    metrics = {
        'LER': sum(dists)/sum(lens),
        'real_mean_length': np.mean(lens),
        'predicted_mean_length': np.mean(pred_lens)
    }
    metrics_file_path = os.path.join(model_path, 'metrics1.json')
    write_dict_to_file(metrics_file_path, metrics)

if __name__ == "__main__":
	main(sys.argv[1], sys.argv[2])