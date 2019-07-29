from keras import layers
from keras import models
from keras import backend as K

def get_residual_block(input_layer, bn=False):
	layer = layers.Conv1D(filters=256, kernel_size=1, strides=1, use_bias=False, padding='same')(input_layer)
	layer = layers.BatchNormalization()(layer)
	layer = layers.ReLU()(layer)
	layer = layers.Conv1D(filters=256, kernel_size=3, strides=1, use_bias=False, padding='same')(layer)
	layer = layers.BatchNormalization()(layer)
	layer = layers.ReLU()(layer)
	layer = layers.Conv1D(filters=256, kernel_size=1, strides=1, use_bias=False, padding='same')(layer)
	layer = layers.BatchNormalization()(layer)
	jump = layers.Conv1D(filters=256, kernel_size=1, strides=1, padding='same')(input_layer)
	if bn:
		jump = layers.BatchNormalization()(jump)
	sum_layer =  layers.Add()([layer, jump])
	sum_layer = layers.ReLU()(sum_layer)
	return sum_layer

def get_RNN_part(input):
	layer_fw = layers.LSTM(units=100, activation='relu',return_sequences=True)(input)
	layer_bw = layers.LSTM(units=100, activation='relu', go_backwards=True,return_sequences=True)(input)
	merged = layers.Concatenate()([layer_fw, layer_bw])
	layer_fw = layers.LSTM(units=100, activation='relu',return_sequences=True)(merged)
	layer_bw = layers.LSTM(units=100, activation='relu', go_backwards=True,return_sequences=True)(merged)
	merged = layers.Concatenate()([layer_fw, layer_bw])
	layer_fw = layers.LSTM(units=100, activation='relu',return_sequences=True)(merged)
	layer_bw = layers.LSTM(units=100, activation='relu', go_backwards=True,return_sequences=True)(merged)
	merged = layers.Concatenate()([layer_fw, layer_bw])
	return merged

def get_GRU_part(input):
	layer_fw = layers.GRU(units=100, activation='relu',return_sequences=True)(input)
	layer_bw = layers.GRU(units=100, activation='relu', go_backwards=True,return_sequences=True)(input)
	merged = layers.Concatenate()([layer_fw, layer_bw])
	layer_fw = layers.GRU(units=100, activation='relu',return_sequences=True)(merged)
	layer_bw = layers.GRU(units=100, activation='relu', go_backwards=True,return_sequences=True)(merged)
	merged = layers.Concatenate()([layer_fw, layer_bw])
	layer_fw = layers.GRU(units=100, activation='relu',return_sequences=True)(merged)
	layer_bw = layers.GRU(units=100, activation='relu', go_backwards=True,return_sequences=True)(merged)
	merged = layers.Concatenate()([layer_fw, layer_bw])
	return merged

def ctc_lambda_func(args):
	y_pred, labels, input_length, label_length = args
	return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def get_default_model():
	input_var = layers.Input(shape=(None, 1), name='the_input')
	input_length = layers.Input(name='input_length', shape=[1], dtype='int64')
	label_length = layers.Input(name='label_length', shape=[1], dtype='int64')
	labels = layers.Input(name='the_labels', shape=[300], dtype='float32')
	model = get_residual_block(input_var)
	for i in range(3):
		model = get_residual_block(model, bn=(i==0))
	model = get_GRU_part(model)
	model = layers.Dense(5)(model)
	model = layers.Activation('softmax')(model)
	loss_out = layers.Lambda(
		ctc_lambda_func, output_shape=(1,),
		name='ctc')([model, labels, input_length, label_length])
	return models.Model(inputs=[input_var, labels, input_length, label_length], outputs=loss_out)
	




