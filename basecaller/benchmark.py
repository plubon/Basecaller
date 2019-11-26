import tensorflow as tf
import os
import sys
import time
from datetime import datetime
from utils import get_first_free_device
import numpy as np
from reader import ChironFileReader
from assembler import AssemblerFactory
from utils import int_label_to_string

eval_batch_size = 1000

def log_time(path, name):
    now = datetime.fromtimestamp(time.time()).isoformat()
    with open(os.path.join(path, 'benchmark.csv'), 'a+') as out:
        out.write(f"{name};{now}\n")


def run(output_path,
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
    log_time(output_path, 'eval_start')
    eval_assemble(output_path, raw_data_path, os.path.join(output_path, 'benchmark'))
    log_time(output_path, 'eval_end')
    print('Finished eval', flush=True)


def eval_assemble(model_dir, data_dir, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    reader = ChironFileReader(data_dir)
    logits_input = tf.placeholder(tf.float32, shape=(None, 300, 5), name='d1')
    len_input = tf.placeholder(tf.int32, shape=(None,), name='d2')
    decoded = tf.nn.ctc_beam_search_decoder(tf.transpose(logits_input, perm=[1, 0, 2], name='d4'),
                                            tf.cast(len_input, tf.int32, name='d3'), merge_repeated=False)[0][0]
    decoded_out = tf.sparse.to_dense(decoded, default_value=-1, name='d6')
    assembler = AssemblerFactory.get('simple')
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        tf.saved_model.loader.load(sess, ["serve"], os.path.join(model_dir, 'saved_model'))
        i = 1
        logits_list = []
        log_time(model_dir, 'setup end')
        signal, indices = reader.read_for_eval(data_dir)
        signal = np.expand_dims(np.stack(signal).astype(np.float32), -1)
        lengths = np.repeat(300, len(indices))
        i = i + 1
        batch_index = 0
        log_time(model_dir, 'file reading end')
        while batch_index < signal.shape[0]:
            print(f'{batch_index}/{signal.shape[0]}',flush=True)
            feed_dict = {
                'IteratorGetNext:0': signal[batch_index:batch_index + eval_batch_size, :, :],
                'IteratorGetNext:2': lengths[batch_index:batch_index + eval_batch_size]
            }
            logits = sess.run('dense/BiasAdd:0', feed_dict=feed_dict)
            logits_list.append(logits)
            batch_index = batch_index + eval_batch_size
        log_time(model_dir, 'network end')
        file_logits = np.concatenate(logits_list, axis=0)
        log_time(model_dir, 'concat end')
        size = file_logits.shape[0]
        decoded = sess.run(decoded_out, feed_dict={
            logits_input: file_logits,
            len_input: np.full(size, 300)})
        log_time(model_dir, 'd  ecode end')
        assembled = assembler.assemble(decoded)
        log_time(model_dir, 'assemble end')
        predicted_seq = int_label_to_string(np.argmax(assembled, axis=0))
        out_filename = f"{'test'}.fasta"
        with open(os.path.join(out_dir, out_filename), 'w') as out_file:
            out_file.write(predicted_seq)
        log_time(model_dir, 'save end')


if __name__ == "__main__":
    run(sys.argv[1],
        sys.argv[2])