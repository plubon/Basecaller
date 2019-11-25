from model import ModelFactory
import tensorflow as tf
import sys

if __name__ == "__main__":
    placeholder = tf.placeholder(tf.float32, [None, 300, 1])
    model = ModelFactory.get(sys.argv[1], None)
