import tensorflow as tf


class OptimizerFactory:

    @staticmethod
    def get(name, logits, labels, seq_length, params=None):
        if params is None:
            params = {}
        if name.lower() == 'adam':
            return AdamOptimizer(logits, labels, seq_length, params)


class AdamOptimizer:

    def __init__(self, logits, labels, seq_length, params):
        self.logits = logits
        self.labels = labels
        self.seq_len = seq_length
        self.loss = tf.reduce_sum(tf.nn.ctc_loss(self.labels, self.logits, tf.cast(self.seq_len,
                                                                                   dtype=tf.int32), time_major=False))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)
