import tensorflow as tf


class ModelFactory:

    @staticmethod
    def get(name, signal, params=None):
        if params is None:
            params = {}
        if name.lower() == 'cnn_lstm':
            return CnnLstmModel(signal, params)
        if name.lower() == 'fcnn_lstm':
            return FcnnLstmModel(signal, params)
        if name.lower() == 'residual':
            return ResidualModel(signal, params)
        if name.lower() == 'tcn':
            return TcnModel(signal, params)
        if name.lower() == 'cnn_tcn':
            return CnnTcnModel(signal, params)
        if name.lower() == 'wavenet':
            return WavenetModel(signal, params)
        else:
            raise ValueError(f'No model with name: {name}')


class WavenetModel:

    def __init__(self, signal, params):
        self.input = signal
        self.params = params
        max_dilation = 128
        skip_connections = []
        i = 1
        model = tf.keras.layers.Conv1D(256, 1, padding='same')(signal)
        while i <= max_dilation:
            model, skip = self.get_wavenet_block(model, params)
            skip_connections.append(skip)
        skip_sum = tf.keras.layers.Add()(skip_connections)
        self.logits = tf.keras.layers.Dense(5)(skip_sum)

    def get_wavenet_block(self, input, dilation, params):
        tanh = tf.keras.layers.Conv1D(256,
                                      2,
                                      dilation_rate=dilation,
                                      padding='causal',
                                      activation='tanh')
        sigma = tf.keras.layers.Conv1D(256,
                                       2,
                                       dilation_rate=dilation,
                                       padding='causal',
                                       activation='sigmoid')
        model = tf.keras.layers.Multiply()([tanh, sigma])
        res = tf.keras.layers.Conv1D(256, 1, padding='same')(model)
        skip = tf.keras.layers.Conv1D(256, 1, padding='same')(model)
        res = tf.keras.layers.Add()[(input, res)]
        return res, skip


class TcnModel:

    def __init__(self, signal, params):
        self.input = signal
        self.params = params

        model = self.get_residual_block(signal, bn=True)
        for i in range(4):
            model = self.get_residual_block(model)

        max_dilation = 32
        i = 1
        while i <= max_dilation:
            model = self.get_block(model, i)
            i = i * 2
        self.logits = tf.keras.layers.Dense(5)(model)

    def get_block(self, input, dilation):
        model = input
        for _ in range(2):
            model = tf.keras.layers.Conv1D(filters=256,
                                           kernel_size=3,
                                           dilation_rate=dilation,
                                           padding='causal')(model)
            model = tf.keras.layers.BatchNormalization()(model)
            model = tf.keras.layers.ReLU()(model)
        jump = tf.keras.layers.Conv1D(filters=256,
                                      kernel_size=1,
                                      padding='same')(input)
        model = tf.keras.layers.Add()([jump, model])
        return tf.keras.layers.ReLU()(model)


class CnnTcnModel:

    def __init__(self, signal, params):
        self.input = signal
        self.params = params

        model = self.get_residual_block(signal, bn=True)
        for i in range(4):
            model = self.get_residual_block(model)

        skip_connections = []
        max_dilation = 32
        i = 1
        while i <= max_dilation:
            jump, model = self.get_block(model, i)
            skip_connections.append(jump)
            i = i * 2
        model = tf.keras.layers.Add()(skip_connections)
        self.logits = tf.keras.layers.Dense(5)(model)

    def get_residual_block(self, input_layer, bn=False):
        layer = tf.keras.layers.Conv1D(filters=256, kernel_size=1, strides=1, use_bias=False, padding='same')(
            input_layer)
        layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.ReLU()(layer)
        layer = tf.keras.layers.Conv1D(filters=256, kernel_size=3, strides=1, use_bias=False, padding='same')(layer)
        layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.ReLU()(layer)
        layer = tf.keras.layers.Conv1D(filters=256, kernel_size=1, strides=1, use_bias=False, padding='same')(layer)
        layer = tf.keras.layers.BatchNormalization()(layer)
        jump = tf.keras.layers.Conv1D(filters=256, kernel_size=1, strides=1, padding='same', use_bias=False)(
            input_layer)
        if bn:
            jump = tf.keras.layers.BatchNormalization()(jump)
        sum_layer = tf.keras.layers.Add()([layer, jump])
        sum_layer = tf.keras.layers.ReLU()(sum_layer)
        return sum_layer

    def get_block(self, input, dilation):
        model = input
        for _ in range(2):
            model = tf.keras.layers.Conv1D(filters=256,
                                           kernel_size=3,
                                           dilation_rate=dilation,
                                           padding='causal')(model)
            model = tf.keras.layers.BatchNormalization()(model)
            model = tf.keras.layers.ReLU()(model)
        jump = tf.keras.layers.Conv1D(filters=256,
                                      kernel_size=1,
                                      padding='same')(input)
        jump = tf.keras.layers.Add()([model, jump])
        jump = tf.keras.layers.ReLU()(jump)
        return jump, model


class ResidualModel:

    def __init__(self, signal, params):
        self.input = signal
        self.params = params

        model = self.get_residual_block(signal, bn=True)
        for i in range(7):
            model = self.get_residual_block(model)

        self.logits = tf.keras.layers.Dense(5)(model)

    def get_residual_block(self, input_layer, bn=False):
        layer = tf.keras.layers.Conv1D(filters=256, kernel_size=1, strides=1, use_bias=False, padding='same')(
            input_layer)
        layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.ReLU()(layer)
        layer = tf.keras.layers.Conv1D(filters=256, kernel_size=3, strides=1, use_bias=False, padding='same')(layer)
        layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.ReLU()(layer)
        layer = tf.keras.layers.Conv1D(filters=256, kernel_size=1, strides=1, use_bias=False, padding='same')(layer)
        layer = tf.keras.layers.BatchNormalization()(layer)
        jump = tf.keras.layers.Conv1D(filters=256, kernel_size=1, strides=1, padding='same', use_bias=False)(
            input_layer)
        if bn:
            jump = tf.keras.layers.BatchNormalization()(jump)
        sum_layer = tf.keras.layers.Add()([layer, jump])
        sum_layer = tf.keras.layers.ReLU()(sum_layer)
        return sum_layer


class FcnnLstmModel:

    def __init__(self, signal, params):
        self.input = signal
        self.params = params

        right = tf.keras.layers.Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(self.input)
        right = tf.keras.layers.BatchNormalization()(right)
        right = tf.keras.layers.Activation('relu')(right)

        right = tf.keras.layers.Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(right)
        right = tf.keras.layers.BatchNormalization()(right)
        right = tf.keras.layers.Activation('relu')(right)

        right = tf.keras.layers.Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(right)
        right = tf.keras.layers.BatchNormalization()(right)
        right = tf.keras.layers.Activation('relu')(right)

        self.logits = tf.keras.layers.Dense(5)(right)


class CnnLstmModel:

    def __init__(self, signal, params):
        self.input = signal
        self.params = params
        model = self.get_residual_block(self.input)
        for i in range(2):
            model = self.get_residual_block(model, bn=(i == 0))
        model = self.get_gru_part(model)
        model = tf.keras.layers.Dense(5)(model)
        self.logits = model

    def get_residual_block(self, input_layer, bn=False):
        layer = tf.keras.layers.Conv1D(filters=256, kernel_size=1, strides=1, use_bias=False, padding='same')(
            input_layer)
        layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.ReLU()(layer)
        layer = tf.keras.layers.Conv1D(filters=256, kernel_size=3, strides=1, use_bias=False, padding='same')(layer)
        layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.ReLU()(layer)
        layer = tf.keras.layers.Conv1D(filters=256, kernel_size=1, strides=1, use_bias=False, padding='same')(layer)
        layer = tf.keras.layers.BatchNormalization()(layer)
        jump = tf.keras.layers.Conv1D(filters=256, kernel_size=1, strides=1, padding='same', use_bias=False)(
            input_layer)
        if bn:
            jump = tf.keras.layers.BatchNormalization()(jump)
        sum_layer = tf.keras.layers.Add()([layer, jump])
        sum_layer = tf.keras.layers.ReLU()(sum_layer)
        return sum_layer

    def get_gru_part(self, input):
        layer_fw = tf.keras.layers.GRU(units=100, activation='relu', return_sequences=True)(input)
        layer_bw = tf.keras.layers.GRU(units=100, activation='relu', go_backwards=True, return_sequences=True)(input)
        merged = tf.keras.layers.Concatenate()([layer_fw, layer_bw])
        layer_fw = tf.keras.layers.GRU(units=100, activation='relu', return_sequences=True)(merged)
        layer_bw = tf.keras.layers.GRU(units=100, activation='relu', go_backwards=True, return_sequences=True)(merged)
        merged = tf.keras.layers.Concatenate()([layer_fw, layer_bw])
        layer_fw = tf.keras.layers.GRU(units=100, activation='relu', return_sequences=True)(merged)
        layer_bw = tf.keras.layers.GRU(units=100, activation='relu', go_backwards=True, return_sequences=True)(merged)
        merged = tf.keras.layers.Concatenate()([layer_fw, layer_bw])
        return merged
