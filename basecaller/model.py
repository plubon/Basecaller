import tensorflow as tf
import blocks
import numpy as np


class ModelFactory:

    @staticmethod
    def get(name, signal, config=None):
        if config is None:
            config = {}
        if name.lower() == 'weird_wavenet':
            return WeirdWavenetModel(signal, config)
        if name.lower() == 'deep_cnn_lstm_identity':
            return DeepCnnLstmIdentityModel(signal, config)
        if name.lower() == 'cnn_lstm':
            return CnnLstmModel(signal, config)
        if name.lower() == 'residual':
            return ResidualModel(signal, config)
        if name.lower() == 'tcn':
            return TcnModel(signal, config)
        if name.lower() == 'cnn_tcn':
            return CnnTcnModel(signal, config)
        if name.lower() == 'cnn_tcn_bidirectional':
            return CnnTcnModelBothDirections(signal, config)
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
        if name.lower() == 'cnn_lstm_identity':
            return CnnLstmIdentityModel(signal, config)
        if name.lower() == 'tcn_identity':
            return CnnTcnIdentityModel(signal, config)
        if name.lower() == 'wavenet_identity':
            return WavenetIdentityModel(signal, config)
        if name.lower() == 'cnn_lstm_dense':
            return CnnLstmDenseModel(signal, config)
        if name.lower() == 'tcn_dense':
            return CnnTcnDenseModel(signal, config)
        if name.lower() == 'wavenet_dense':
            return WavenetDenseModel(signal, config)
        else:
            raise ValueError(f'No model with name: {name}')


class WavenetDenseModel:

    def __init__(self, signal, config):
        growth_rate = 15
        self.input = signal
        self.config = config
        features = signal
        for i in range(11):
            cb = tf.keras.layers.BatchNormalization()(features)
            cb = tf.keras.layers.ReLU()(cb)
            cb = tf.keras.layers.Conv1D(filters=growth_rate, kernel_size=3, use_bias=False, padding='same')(cb)
            features = tf.concat([features, cb], axis=-1)
        max_dilation = 64
        dilation = 1
        while dilation <= max_dilation:
            reversed = tf.reverse(features, [1])
            original_branch = blocks.wavenet_gate(features, dilation, 10)
            reversed_branch = blocks.wavenet_gate(reversed, dilation, 10)
            features = tf.concat([features, original_branch, reversed_branch], axis=-1)
            dilation = dilation * 2
        model = tf.keras.layers.Dense(5)(features)
        self.logits = model


class WeirdWavenetModel:

    def __init__(self, signal, params):
        self.input = signal
        self.params = params
        max_dilation = 128
        model = tf.keras.layers.Conv1D(filters=256, kernel_size=1, strides=1, use_bias=False, padding='same')(signal)
        for i in range(12):
            model = blocks.pre_activation_residual_block(signal)
        i = 1
        while i <= max_dilation:
            model, _ = blocks.wavenet_weird_block(model, i)
            model, _ = blocks.wavenet_weird_block(model, i)
            i = i * 2
        self.logits = tf.keras.layers.Dense(5)(model)

class WavenetIdentityModel:

    def __init__(self, signal, params):
        self.input = signal
        self.params = params
        max_dilation = 128
        model = tf.keras.layers.Conv1D(filters=256, kernel_size=1, strides=1, use_bias=False, padding='same')(signal)
        for i in range(5):
            model = blocks.pre_activation_residual_block(signal)
        i = 1
        while i <= max_dilation:
            model, _ = blocks.wavenet_identity_block(model, i)
            i = i * 2
        self.logits = tf.keras.layers.Dense(5)(model)


class CnnTcnDenseModel:

    def __init__(self, signal, config):
        growth_rate = 15
        self.input = signal
        self.config = config
        features = signal
        for i in range(11):
            cb = tf.keras.layers.BatchNormalization()(features)
            cb = tf.keras.layers.ReLU()(cb)
            cb = tf.keras.layers.Conv1D(filters=growth_rate, kernel_size=3, use_bias=False, padding='same')(cb)
            features = tf.concat([features, cb], axis=-1)
        max_dilation = 64
        dilation = 1
        while dilation <= max_dilation:
            reversed = tf.reverse(features, [1])
            model = tf.keras.layers.Conv1D(filters=10,
                                           kernel_size=3,
                                           dilation_rate=dilation,
                                           padding='causal')(features)
            reversed = tf.keras.layers.Conv1D(filters=10,
                                              kernel_size=3,
                                              dilation_rate=dilation,
                                              padding='causal')(reversed)
            features = tf.concat([features, model, reversed], axis=-1)
            dilation = dilation * 2
        model = tf.keras.layers.Dense(5)(features)
        self.logits = model


class CnnTcnIdentityModel:

    def __init__(self, signal, config):
        self.input = signal
        self.config = config
        model = tf.keras.layers.Conv1D(filters=256, kernel_size=1, strides=1, use_bias=False, padding='same')(signal)
        for i in range(5):
            model = blocks.pre_activation_residual_block(model)
        max_dilation = 64
        i = 1
        while i <= max_dilation:
            model = blocks.tcn_identity_block(model, i)
            i = i * 2
        self.logits = tf.keras.layers.Dense(5)(model)


class CnnLstmDenseModel:
    def __init__(self, signal, config):
        growth_rate = 15
        self.input = signal
        self.config = config
        features = signal
        for i in range(11):
            cb = tf.keras.layers.BatchNormalization()(features)
            cb = tf.keras.layers.ReLU()(cb)
            cb = tf.keras.layers.Conv1D(filters=growth_rate, kernel_size=3, use_bias=False, padding='same')(cb)
            features = tf.concat([features, cb], axis=-1)
        for _ in range(3):
            cb = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(growth_rate, return_sequences=True), merge_mode='concat')(features)
            features = tf.concat([features, cb], axis=-1)
        model = tf.keras.layers.Dense(5)(features)
        self.logits = model


class DeepCnnLstmIdentityModel:
    def __init__(self, signal, config):
        hidden_num = 128
        self.input = signal
        self.config = config
        model = tf.keras.layers.Conv1D(filters=256, kernel_size=1, strides=1, use_bias=False, padding='same')(signal)
        for i in range(12):
            model = blocks.pre_activation_residual_block(model)
        for _ in range(5):
            lstm = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(128, return_sequences=True), merge_mode='concat')(model)
            model = tf.keras.layers.Add()([lstm, model])
        weight_bi = tf.Variable(tf.truncated_normal([2, hidden_num], stddev=np.sqrt(2.0 / (2*hidden_num))))
        bias_bi = tf.Variable(tf.zeros([hidden_num]))
        model = tf.reshape(model, [tf.shape(model)[0], 300, 2, hidden_num])
        model = tf.nn.bias_add(tf.reduce_sum(tf.multiply(model, weight_bi), axis=2), bias_bi)
        model = tf.keras.layers.Dense(5)(model)
        self.logits = model


class CnnLstmIdentityModel:
    def __init__(self, signal, config):
        hidden_num = 128
        self.input = signal
        self.config = config
        model = tf.keras.layers.Conv1D(filters=256, kernel_size=1, strides=1, use_bias=False, padding='same')(signal)
        for i in range(5):
            model = blocks.pre_activation_residual_block(model)
        model = blocks.lstm_identity_block(model)
        weight_bi = tf.Variable(tf.truncated_normal([2, hidden_num], stddev=np.sqrt(2.0 / (2*hidden_num))))
        bias_bi = tf.Variable(tf.zeros([hidden_num]))
        model = tf.reshape(model, [tf.shape(model)[0], 300, 2, hidden_num])
        model = tf.nn.bias_add(tf.reduce_sum(tf.multiply(model, weight_bi), axis=2), bias_bi)
        model = tf.keras.layers.Dense(5)(model)
        self.logits = model

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
        model = tf.keras.layers.Conv1D(filters=256, padding='same', kernel_size=1)(model)
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
            jump, model = blocks.tcn_block_both_directions(model, i)
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
