import tensorflow as tf
import blocks
import numpy as np


class ModelFactory:

    @staticmethod
    def get(name, signal, config=None):
        if config is None:
            config = {}
        if name.lower() == 'cnn_lstm':
            return CnnLstmModel(signal, config)
        if name.lower() == 'residual':
            return ResidualModel(signal, config)
        if name.lower() == 'tcn':
            return TcnModel(signal, config)
        if name.lower() == 'cnn_tcn':
            return CnnTcnModel(signal, config)
        if name.lower() == 'wavenet':
            return WavenetModel(signal, config)
        if name.lower() == 'wavenet_bidirectional':
            return WavenetBidirectionalModel(signal, config)
        if name.lower() == 'placeholder':
            return PlaceholderModel(signal, config)
        if name.lower() == 'wavenet_pre':
            return WavenetPreActivationModel(signal, config)
        if name.lower() == 'dense_net':
            return DenseNetModel(signal, config)
        if name.lower() == 'dense_net_lstm':
            return DenseNetLstmModel(signal, config)
        if name.lower() == 'dense_wave_net':
            return DenseWaveNetModel(signal, config)
        else:
            raise ValueError(f'No model with name: {name}')


class PlaceholderModel:

    def __init__(self, signal, params):
        self.input = signal
        self.params = params
        self.logits = signal


class DenseNetLstmModel:

    def __init__(self, signal, params):
        self.input = signal
        self.params = params
        hidden_num = 100
        model = blocks.dense_net(self.input)
        model = blocks.lstm_block(model)
        weight_bi = tf.Variable(tf.truncated_normal([2, hidden_num], stddev=np.sqrt(2.0 / (2 * hidden_num))))
        bias_bi = tf.Variable(tf.zeros([hidden_num]))
        model = tf.reshape(model, [tf.shape(model)[0], 300, 2, hidden_num])
        model = tf.nn.bias_add(tf.reduce_sum(tf.multiply(model, weight_bi), axis=2), bias_bi)
        model = tf.keras.layers.Dense(5)(model)
        self.logits = model


class DenseWaveNetModel:

    def __init__(self, signal, params):
        self.input = signal
        self.params = params
        model = blocks.dense_net(self.input)
        max_dilation = 128
        i = 1
        skip_connections = []
        while i <= max_dilation:
            model, skip = blocks.wavenet_bidirectional_block(model, i)
            skip_connections.append(skip)
            i = i * 2
        skip_sum = tf.keras.layers.Add()(skip_connections)
        self.logits = tf.keras.layers.Dense(5)(skip_sum)


class DenseNetModel:

    def __init__(self, signal, params):
        self.input = signal
        self.params = params
        model = blocks.dense_net(self.input)
        self.logits = tf.keras.layers.Dense(5)(model)


class WavenetPreActivationModel:

    def __init__(self, signal, params):
        self.input = signal
        self.params = params
        max_dilation = 128
        model = tf.keras.layers.Conv1D(filters=256, kernel_size=1, strides=1, use_bias=False, padding='same')(signal)
        for i in range(5):
            model = blocks.pre_activation_residual_block(model)
        i = 1
        skip_connections = []
        while i <= max_dilation:
            model, skip = blocks.wavenet_bidirectional_block(model, i)
            skip_connections.append(skip)
            i = i * 2
        skip_sum = tf.keras.layers.Add()(skip_connections)
        self.logits = tf.keras.layers.Dense(5)(skip_sum)


class WavenetModel:

    def __init__(self, signal, params):
        self.input = signal
        self.params = params
        model = self.input
        for i in range(3):
            model = blocks.residual_block(model, i == 0)
        max_dilation = 128
        skip_connections = []
        i = 1
        while i <= max_dilation:
            model, skip = blocks.wavenet_block(model, i, params)
            skip_connections.append(skip)
            i = i * 2
        skip_sum = tf.keras.layers.Add()(skip_connections)
        self.logits = tf.keras.layers.Dense(5)(skip_sum)


class WavenetBidirectionalModel:

    def __init__(self, signal, params):
        self.input = signal
        self.params = params
        max_dilation = 128
        model = self.input
        for i in range(3):
            model = blocks.residual_block(model, i == 0)
        i = 1
        skip_connections = []
        while i <= max_dilation:
            model, skip = blocks.wavenet_bidirectional_block(model, i)
            skip_connections.append(skip)
            i = i * 2
        skip_sum = tf.keras.layers.Add()(skip_connections)
        self.logits = tf.keras.layers.Dense(5)(skip_sum)


class TcnModel:

    def __init__(self, signal, params):
        self.input = signal
        self.params = params

        model = blocks.residual_block(signal, bn=True)
        for i in range(4):
            model = blocks.residual_block(model)

        max_dilation = 64
        i = 1
        while i <= max_dilation:
            model = blocks.tcn_block(model, i)
            i = i * 2
        self.logits = tf.keras.layers.Dense(5)(model)


class CnnTcnModel:

    def __init__(self, signal, params):
        self.input = signal
        self.params = params

        model = blocks.residual_block(signal, bn=True)
        for i in range(4):
            model = blocks.residual_block(model)

        skip_connections = []
        max_dilation = 64
        i = 1
        while i <= max_dilation:
            jump, model = blocks.tcn_block(model, i)
            skip_connections.append(jump)
            i = i * 2
        model = tf.keras.layers.Add()(skip_connections)
        self.logits = tf.keras.layers.Dense(5)(model)


class CnnTcnModelBothDirections:

    def __init__(self, signal, params):
        self.input = signal
        self.params = params

        model = blocks.residual_block(signal, bn=True)
        for i in range(4):
            model = blocks.residual_block(model)

        skip_connections = []
        max_dilation = 64
        i = 1
        while i <= max_dilation:
            jump, model = blocks.tcn_block(model, i)
            skip_connections.append(jump)
            i = i * 2
        model = tf.keras.layers.Add()(skip_connections)
        self.logits = tf.keras.layers.Dense(5)(model)


class ResidualModel:

    def __init__(self, signal, params):
        self.input = signal
        self.params = params

        model = blocks.residual_block(signal, bn=True)
        for i in range(7):
            model = blocks.residual_block(model)

        self.logits = tf.keras.layers.Dense(5)(model)


class CnnLstmModel:

    def __init__(self, signal, config):
        hidden_num = 100
        self.input = signal
        self.config = config
        model = signal
        for i in range(5):
            model = blocks.pre_activation_residual_block(model)
        model = blocks.lstm_block(model)
        weight_bi = tf.Variable(tf.truncated_normal([2, hidden_num], stddev=np.sqrt(2.0 / (2*hidden_num))))
        bias_bi = tf.Variable(tf.zeros([hidden_num]))
        model = tf.reshape(model, [tf.shape(model)[0], 300, 2, hidden_num])
        model = tf.nn.bias_add(tf.reduce_sum(tf.multiply(model, weight_bi), axis=2), bias_bi)
        model = tf.keras.layers.Dense(5)(model)
        self.logits = model
