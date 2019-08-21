import tensorflow as tf


class OptimizerFactory:

    @staticmethod
    def get(name, logits, labels, seq_length, params=None):
        if params is None:
            params = {}
        if name.lower() == 'adam_clipped':
            return ClippedAdamOptimizer(logits, labels, seq_length, params)
        if name.lower() == 'adam':
            return AdamOptimizer(logits, labels, seq_length, params)


class ClippedAdamOptimizer:

    def __init__(self, logits, labels, seq_length, params):
        self.logits = logits
        self.labels = labels
        self.seq_len = seq_length
        self.loss = tf.reduce_mean(tf.nn.ctc_loss(self.labels, self.logits, tf.cast(self.seq_len,
                                                                                   dtype=tf.int32), time_major=False))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0005)
        gds, vars = zip(*self.optimizer.compute_gradients(self.loss))
        clipped_gds, _ = tf.clip_by_global_norm(gds, 5)
        self.optimizer = self.optimizer.apply_gradients(zip(clipped_gds, vars))


class AdamOptimizer:

    def __init__(self, logits, labels, seq_length, params):
        self.logits = logits
        self.labels = labels
        self.seq_len = seq_length
        self.loss = tf.reduce_mean(tf.nn.ctc_loss(self.labels, self.logits, tf.cast(self.seq_len,
                                                                                   dtype=tf.int32), time_major=False))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)
