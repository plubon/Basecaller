from keras import layers
from keras import models
from keras import backend as K

def get_residual_block(input_layer):
	layer = layers.Conv1D(filters=256, kernel_size=1, strides=1, use_bias=False, padding='same')(input_layer)
	layer = layers.ReLU()(layer)
	layer = layers.Conv1D(filters=256, kernel_size=3, strides=1, use_bias=False, padding='same')(layer)
	layer = layers.ReLU()(layer)
	layer = layers.Conv1D(filters=256, kernel_size=1, strides=1, use_bias=False, padding='same')(layer)
	jump = layers.Conv1D(filters=256, kernel_size=1, strides=1, use_bias=False, padding='same')(input_layer)
	sum_layer =  layers.Add()([layer, jump])
	sum_layer = layers.BatchNormalization()(sum_layer)
	sum_layer = layers.ReLU()(sum_layer)
	return sum_layer

def get_GRU_part(input):
	layer_fw = layers.GRU(units=200, activation='relu',return_sequences=True)(input)
	layer_bw = layers.GRU(units=200, activation='relu', go_backwards=True,return_sequences=True)(input)
	merged = layers.Concatenate()([layer_fw, layer_bw])
	layer_fw = layers.GRU(units=200, activation='relu',return_sequences=True)(merged)
	layer_bw = layers.GRU(units=200, activation='relu', go_backwards=True,return_sequences=True)(merged)
	merged = layers.Concatenate()([layer_fw, layer_bw])
	layer_fw = layers.GRU(units=200, activation='relu',return_sequences=True)(merged)
	layer_bw = layers.GRU(units=200, activation='relu', go_backwards=True,return_sequences=True)(merged)
	merged = layers.Concatenate()([layer_fw, layer_bw])
	return merged

def get_LSTM_part(input):
	model = layers.Bidirectional(layers.LSTM(units=200, activation='relu',return_sequences=True), merge_mode='concat')(input)
	model = layers.Bidirectional(layers.LSTM(units=200, activation='relu',return_sequences=True), merge_mode='concat')(model)
	model = layers.Bidirectional(layers.LSTM(units=200, activation='relu',return_sequences=True), merge_mode='concat')(model)
	return model

def ctc_lambda_func(args):
	y_pred, labels, input_length, label_length = args
	return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def get_default_model():
	input_var = layers.Input(shape=(None, 1), name='the_input')
	input_length = layers.Input(name='input_length', shape=[1], dtype='int64')
	label_length = layers.Input(name='label_length', shape=[1], dtype='int64')
	labels = layers.Input(name='the_labels', shape=[300], dtype='float32')
	model = get_residual_block(input_var)
	for _ in range(4):
		model = get_residual_block(model)
	model = get_LSTM_part(model)
	model = layers.Dense(5)(model)
	model = layers.Activation('softmax', name='softmax')(model)
	loss_out = layers.Lambda(
		ctc_lambda_func, output_shape=(1,),
		name='ctc')([model, labels, input_length, label_length])
	return models.Model(inputs=[input_var, labels, input_length, label_length], outputs=loss_out)

def get_model_with_boundaries():
	input_var = layers.Input(shape=(None, 1), name='the_input')
	input_length = layers.Input(name='input_length', shape=[1], dtype='int64')
	label_length = layers.Input(name='label_length', shape=[1], dtype='int64')
	labels = layers.Input(name='the_labels', shape=[300], dtype='float32')
	boundaries = layers.Input(shape=(None, 1), name='the_boundaries')
	model = layers.Concatenate()([input_var, boundaries])
	model = get_residual_block(model)
	for _ in range(4):
		model = get_residual_block(model)
	model = get_LSTM_part(model)
	model = layers.Dense(5)(model)
	model = layers.Concatenate()([model, boundaries])
	model = layers.Activation('softmax', name='softmax')(model)
	loss_out = layers.Lambda(
		ctc_lambda_func, output_shape=(1,),
		name='ctc')([model, labels, input_length, label_length])
	return models.Model(inputs=[input_var, boundaries, labels, input_length, label_length], outputs=loss_out)
	


	




