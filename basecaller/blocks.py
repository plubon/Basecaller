import tensorflow as tf


def wavenet_block(input, dilation, params):
    tanh = tf.keras.layers.Conv1D(256,
                                  2,
                                  dilation_rate=dilation,
                                  padding='causal',
                                  activation='tanh')(input)
    sigma = tf.keras.layers.Conv1D(256,
                                   2,
                                   dilation_rate=dilation,
                                   padding='causal',
                                   activation='sigmoid')(input)
    model = tf.keras.layers.Multiply()([tanh, sigma])
    res = tf.keras.layers.Conv1D(256, 1, padding='same')(model)
    skip = tf.keras.layers.Conv1D(256, 1, padding='same')(model)
    res = tf.keras.layers.Add()([input, res])
    return res, skip


def tcn_block(input, dilation):
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


def tcn_block_both_direction(original, reversed, dilation):
    for _ in range(2):
        conv = tf.keras.layers.Conv1D(filters=256,
                                      kernel_size=3,
                                      dilation_rate=dilation,
                                      padding='causal')
        original = conv(original)
        reversed = conv(reversed)
        bn = tf.keras.layers.BatchNormalization()
        original = bn(original)
        reversed = bn(reversed)
        activation = tf.keras.layers.ReLU()
        original = activation(original)
        reversed = activation(reversed)
    jump = tf.keras.layers.Conv1D(filters=256,
                                  kernel_size=1,
                                  padding='same')
    jump_original = jump(original)
    jump_reversed = jump(reversed)
    original = tf.keras.layers.Add()([jump_original, original])
    reversed = tf.keras.layers.Add()([jump_reversed, reversed])
    relu = tf.keras.layers.ReLU()
    return relu(original), relu(reversed)


def wavenet_block_both_direction(original, reversed, dilation):
    prev_original = original
    prev_reversed = reversed
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
    multiply = tf.keras.layers.Multiply()
    original = multiply([tanh(original), sigma(original)])
    reversed = multiply([tanh(reversed), sigma(reversed)])
    res = tf.keras.layers.Conv1D(256, 1, padding='same')
    skip = tf.keras.layers.Conv1D(256, 1, padding='same')
    original_res = tf.keras.layers.Add()([prev_original, res(original)])
    reversed_res = tf.keras.layers.Add()([prev_reversed, res(reversed)])
    original_skip = skip(original)
    reversed_skip = skip(reversed)
    return original_res, original_skip, reversed_res, reversed_skip


def residual_block(input_layer, bn=False):
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


def lstm_block(input):
    model = input
    for _ in range(3):
        model = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(100, return_sequences=True), merge_mode='concat')(model)
    return model
