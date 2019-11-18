from assembler import AssemblerFactory
from decoder import DecoderFactory
import tensorflow as tf
import numpy as np
import os
from utils import int_label_to_string
import sys
from reader import ChironFileReader, H5FileReader
from Levenshtein import distance


def assemble(input_path, output_path, decoder, assembler, target_dir=None):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    data_files = [x for x in os.listdir(input_path) if x.endswith('.npy')]
    logits_input = tf.placeholder(tf.float32, shape=(None, 300, 5))
    len_input = tf.placeholder(tf.int32, shape=(None,))
    decoder = DecoderFactory.get(decoder, logits_input, len_input)
    decoded_out = tf.sparse.to_dense(decoder.decoded, default_value=-1)
    assembler = AssemblerFactory.get(assembler)
    distances = []
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        for no, file in enumerate(data_files):
            logits = np.load(os.path.join(input_path, file))
            size = logits.shape[0]
            decoded = sess.run(decoded_out, feed_dict={
                logits_input: logits,
                len_input: np.full(size, 300)})
            assembled = assembler.assemble(decoded)
            out_filename = f"{''.join(file.split('.')[0])}.fastq"
            predicted_seq = int_label_to_string(np.argmax(assembled, axis=0))
            if target_dir is not None:
                target_filename = '.'.join(file.split('.')[:-1]).replace('.signal', '.label')
                if target_filename.endswith('fast5'):
                    reader = H5FileReader()
                else:
                    reader = ChironFileReader()
                target_seq = reader.read_entire_sequence(os.path.join(target_dir, target_filename))
                distances.append((out_filename, distance(target_seq, predicted_seq)/len(target_seq)))
            with open(os.path.join(output_path, out_filename), 'w') as out_file:
                out_file.write(predicted_seq)
            print(f"{no}/{len(data_files)} Processed {file}")
    with open(os.path.join(output_path, 'file_results.csv'),'w') as results_file:
        for result in distances:
            results_file.write(f"{result[0]},{result[1]}\n")


if __name__ == "__main__":
    assemble(sys.argv[1], sys.argv[2], 'beam_search', 'simple', sys.argv[3])
