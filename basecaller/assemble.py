from assembler import AssemblerFactory
from decoder import DecoderFactory
import tensorflow as tf
import numpy as np
import os
from utils import int_label_to_string
import sys


def assemble(input_path, output_path, decoder, assembler):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    data_files = [x for x in os.listdir(input_path) if x.endswith('.npy')]
    logits_input = tf.placeholder(tf.float32, shape=(None, 300, 5))
    len_input = tf.placeholder(tf.int32, shape=(None,))
    decoder = DecoderFactory.get(decoder, logits_input, len_input)
    decoded_out = tf.sparse.to_dense(decoder.decoded, default_value=-1)
    assembler = AssemblerFactory.get(assembler)
    with tf.Session() as sess:
        for file in data_files:
            logits = np.load(os.path.join(input_path, file))
            size = logits.shape[0]
            decoded = sess.run([decoded_out], feed_dict={
                logits_input: logits,
                len_input: np.full(size, 300)})
            assembled = assembler.assemble(decoded)
            out_filename = f"{''.join(file.split('.')[:-1])}.fast5"
            with open(os.path.join(output_path, out_filename), 'w') as out_file:
                out_file.write(int_label_to_string(np.argmax(assembled, axis=0)))


if __name__ == "__main__":
    assemble(sys.argv[1], sys.argv[2], 'beam_search', 'simple')
