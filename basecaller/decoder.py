import tensorflow as tf


class DecoderFactory:

    @staticmethod
    def get(name, logits, seq_length, params=None):
        if params is None:
            params = {}
        if name.lower() == 'beam_search':
            return BeamSearchDecoder(logits, seq_length, params)


class BeamSearchDecoder:

    def __init__(self, logits, seq_length, params):
        self.logits = logits
        self.seq_length = seq_length
        self.decoded = tf.nn.ctc_beam_search_decoder(tf.transpose(self.logits, perm=[1, 0, 2]),
                                                     tf.cast(self.seq_length, tf.int32), merge_repeated=False)[0][0]
