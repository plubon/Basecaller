import tensorflow as tf
import os
from train import train
from test import test
import sys
import time
from datetime import datetime
from utils import get_first_free_device
import numpy as np
from data import EvalDataExtractor
from assembler import AssemblerFactory
from utils import int_label_to_string
from calculate_metrics import calculate
eval_batch_size = 250


def log_time(path, name, time):
    with open(os.path.join(path, 'times.csv'), 'a+') as out:
        out.write(f"{name};{time}\n")


def run(config_path,
        train_dataset_path,
        val_dataset_path,
        output_path,
        raw_data_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    attempts = 0
    device = get_first_free_device()
    while attempts < 5 and device is None:
        print("All devices busy, retrying...", flush=True)
        time.sleep(60)
        device = get_first_free_device()
    print(f"Using device:{device}", flush=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    if not os.path.exists(os.path.join(output_path, 'saved_model')):
        train_start = datetime.fromtimestamp(time.time()).isoformat()
        log_time(output_path, 'train_start', train_start)
        train(config_path, train_dataset_path, val_dataset_path, output_path, early_stop=True)
        train_end = datetime.fromtimestamp(time.time()).isoformat()
        log_time(output_path, 'train_end', train_end)
        print(f'Finished training', flush=True)
        tf.reset_default_graph()
        test(output_path, train_dataset_path)
        tf.reset_default_graph()
    else:
        print("Skipping training since saved model is present", flush=True)
    eval_start = datetime.fromtimestamp(time.time()).isoformat()
    log_time(output_path, 'eval_start', eval_start)
    eval_assemble(output_path, raw_data_path, os.path.join(output_path, 'fasta'))
    eval_end = datetime.fromtimestamp(time.time()).isoformat()
    log_time(output_path, 'eval_end', eval_end)
    print('Finished eval', flush=True)
    calculate(os.path.join(output_path, 'fasta'), output_path)


def eval_assemble(model_dir, data_dir, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    data_extractor = EvalDataExtractor(data_dir)
    data_extractor.filter('coli')
    data_extractor.set_limit(200)
    logits_input = tf.placeholder(tf.float32, shape=(None, 300, 5), name='d1')
    len_input = tf.placeholder(tf.int32, shape=(None,), name='d2')
    decoded = tf.nn.ctc_beam_search_decoder(tf.transpose(logits_input, perm=[1, 0, 2], name='d4'),
                                            tf.cast(len_input, tf.int32, name='d3'), merge_repeated=False, name='d5')[0][0]
    decoded_out = tf.sparse.to_dense(decoded, default_value=-1, name='d6')
    assembler = AssemblerFactory.get('simple')
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        tf.saved_model.loader.load(sess, ["serve"], os.path.join(model_dir, 'saved_model'))
        i = 1
        size = data_extractor.get_size()
        while data_extractor.has_next_file():
            logits_list = []
            signal, lengths, indices, filename = data_extractor.extract_next_file()
            print(f"{i}/{size} : {filename} ")
            i = i + 1
            if len(signal) == 0:
                continue
            batch_index = 0
            while batch_index < signal.shape[0]:
                feed_dict = {
                    'IteratorGetNext:0': signal[batch_index:batch_index + eval_batch_size, :, :],
                    'IteratorGetNext:2': lengths[batch_index:batch_index + eval_batch_size]
                }
                logits = sess.run('dense/BiasAdd:0', feed_dict=feed_dict)
                logits_list.append(logits)
                batch_index = batch_index + eval_batch_size
            file_logits = np.concatenate(logits_list, axis=0)
            size = logits.shape[0]
            decoded = sess.run(decoded_out, feed_dict={
                logits_input: file_logits,
                len_input: np.full(size, 300)})
            assembled = assembler.assemble(decoded)
            predicted_seq = int_label_to_string(np.argmax(assembled, axis=0))
            out_filename = f"{''.join(filename.split('.')[0])}.fasta"
            with open(os.path.join(out_dir, out_filename), 'w') as out_file:
                out_file.write(predicted_seq)
            print(f"Processed file :{filename}", flush=True)
            calculate()


if __name__ == "__main__":
    run(sys.argv[1],
        sys.argv[2],
        sys.argv[3],
        sys.argv[4],
        sys.argv[5])
