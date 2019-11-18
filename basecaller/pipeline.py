import tensorflow as tf
import os
from train import train
from test import test
from eval import evaluate
from assemble import assemble
import sys
import time
from datetime import datetime
from utils import get_first_free_device


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
    with tf.device(f'/device:GPU:{device}'):
        train_start = datetime.fromtimestamp(time.time()).isoformat()
        log_time(output_path, 'train_start', train_start)
        train(config_path, train_dataset_path, val_dataset_path, output_path)
        train_end = datetime.fromtimestamp(time.time()).isoformat()
        log_time(output_path, 'train_end', train_end)
        print(f'Finished training')
        tf.reset_default_graph()
        test(output_path, train_dataset_path)
        tf.reset_default_graph()
        eval_start = datetime.fromtimestamp(time.time()).isoformat()
        log_time(output_path, 'eval_start', eval_start)
        evaluate(output_path, raw_data_path, os.path.join(output_path, 'logits'))
        eval_end = datetime.fromtimestamp(time.time()).isoformat()
        log_time(output_path, 'eval_end', eval_end)
        print('Finished calculating logits')
        tf.reset_default_graph()
        assemble_start = datetime.fromtimestamp(time.time()).isoformat()
        log_time(output_path, 'assemble_start', assemble_start)
        assemble(os.path.join(output_path, 'logits'),
                 os.path.join(output_path, 'fasta'),
                 'beam_search',
                 'simple')
        assemble_end = datetime.fromtimestamp(time.time()).isoformat()
        log_time(output_path, 'assemble_end', assemble_end)
        print('Finished assembly')


if __name__ == "__main__":
    run(sys.argv[1],
        sys.argv[2],
        sys.argv[3],
        sys.argv[4],
        sys.argv[5])
