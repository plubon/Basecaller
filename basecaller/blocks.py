import tensorflow as tf


def wavenet_gate(input, dilation, filters):
    tanh = tf.keras.layers.Conv1D(filters,
                                  2,
                                  dilation_rate=dilation,
                                  padding='causal',
                                  activation='tanh')(input)
    sigma = tf.keras.layers.Conv1D(filters,
                                   2,
                                   dilation_rate=dilation,
                                   padding='causal',
                                   activation='sigmoid')(input)
    return tf.keras.layers.Multiply()([tanh, sigma])


def wavenet_block(input, dilation, params=None):
    model = wavenet_gate(input, dilation, 256)
    res = tf.keras.layers.Conv1D(256, 1, padding='same')(model)
    skip = tf.keras.layers.Conv1D(256, 1, padding='same')(model)
    res = tf.keras.layers.Add()([input, res])
    return res, skip


def wavenet_bidirectional_block(input, dilation):
    reversed = tf.reverse(input, [1])
    original_branch = wavenet_gate(input, dilation, 128)
    reversed_branch = wavenet_gate(reversed, dilation, 128)
    merged = tf.concat([original_branch, tf.reverse(reversed_branch, [1])], axis=-1)
    res = tf.keras.layers.Conv1D(256, 1, padding='same')(merged)
    skip = tf.keras.layers.Conv1D(256, 1, padding='same')(merged)
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


def pre_activation_residual_block(input_layer):
    layer = tf.keras.layers.BatchNormalization()(input_layer)
    layer = tf.keras.layers.ReLU()(layer)
    layer = tf.keras.layers.Conv1D(filters=256, kernel_size=3, strides=1, use_bias=False, padding='same')(layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.keras.layers.ReLU()(layer)
    layer = tf.keras.layers.Conv1D(filters=256, kernel_size=3, strides=1, use_bias=False, padding='same')(layer)
    sum_layer = tf.keras.layers.Add()([layer, input_layer])
    return sum_layer


def lstm_block(input):
    model = input
    for _ in range(3):
        model = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(100, return_sequences=True), merge_mode='concat')(model)
    return model
