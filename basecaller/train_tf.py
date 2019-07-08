import tensorflow as tf
from Data import Dataset, ExampleSequence, TrainingExample
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.rnn.python.ops.rnn import stack_bidirectional_dynamic_rnn
import numpy as np
import sys

batch_size = 150
sequence_len = 400


def get_sparse(batch):
    shape = [batch_size, sequence_len]
    indices = []
    values = []
    lengths = batch['label_length']
    labels = batch['the_labels']
    for b in range(batch_size):
        for t in range(lengths[b]):
            indices.append([b, t])
            values.append(labels[b, t])
    return tf.sparse.SparseTensor(indices, values, shape)


def simple_global_bn(inp,name):
    ksize = inp.get_shape().as_list()
    ksize = [ksize[-1]]
    mean,variance = tf.nn.moments(inp,[0,1,2],name=name+'_moments')
    scale = tf.get_variable(name+"_scale", shape=ksize)
    offset = tf.get_variable(name+"_offset", shape=ksize)
    return tf.nn.batch_normalization(inp,mean=mean,variance=variance,scale=scale,offset=offset,variance_epsilon=1e-5)


def conv_layer(indata,ksize,padding,name,dilate = 1,strides=[1,1,1,1],bias_term = False,active = True,BN= True):
    """A standard convlotional layer"""
    with tf.variable_scope(name):
        W = tf.get_variable("weights", dtype = tf.float32, shape=ksize,initializer=tf.contrib.layers.xavier_initializer())
        if bias_term:
            b = tf.get_variable("bias", dtype=tf.float32,shape=[ksize[-1]])
        if dilate>1:
            if bias_term:
                conv_out = b + tf.nn.atrous_conv2d(indata,W,rate = dilate,padding=padding,name=name)
            else:
                conv_out = tf.nn.atrous_conv2d(indata,W,rate = dilate,padding=padding,name=name)
        else:
            if bias_term:
                conv_out = b + tf.nn.conv2d(indata,W,strides = strides,padding = padding,name = name)
            else:
                conv_out = tf.nn.conv2d(indata,W,strides = strides,padding = padding,name = name)
    if BN:
        with tf.variable_scope(name+'_bn') as scope:
            conv_out = simple_global_bn(conv_out,name = name+'_bn')
    if active:
        with tf.variable_scope(name+'_relu'):
            conv_out = tf.nn.relu(conv_out,name='relu')
    return conv_out


def residual_layer(indata, out_channel, i_bn = False):
    fea_shape = indata.get_shape().as_list()
    in_channel = fea_shape[-1]
    with tf.variable_scope('branch1'):
        indata_cp = conv_layer(indata,ksize = [1,1,in_channel,out_channel],padding = 'SAME', name = 'conv1',BN = i_bn,active = False)
    with tf.variable_scope('branch2'):
        conv_out1 = conv_layer(indata,ksize = [1,1,in_channel,out_channel],padding = 'SAME', name = 'conv2a',bias_term = False)
        conv_out2 = conv_layer(conv_out1,ksize = [1,3,out_channel,out_channel],padding = 'SAME', name = 'conv2b',bias_term = False)
        conv_out3 = conv_layer(conv_out2,ksize = [1,1,out_channel,out_channel],padding = 'SAME', name = 'conv2c',bias_term = False,active = False)
    with tf.variable_scope('plus'):
        relu_out = tf.nn.relu(indata_cp+conv_out3,name = 'final_relu')
    return relu_out


def getcnnfeature(signal):
    signal_shape = signal.get_shape().as_list()
    signal = tf.reshape(signal, [signal_shape[0],1,signal_shape[1],1])
    with tf.variable_scope('res_layer1'):
        res1 = residual_layer(signal,out_channel = 256, i_bn = True)
    with tf.variable_scope('res_layer2'):
        res2 = residual_layer(res1,out_channel = 256)
    with tf.variable_scope('res_layer3'):
        res3 = residual_layer(res2,out_channel = 256)
    feashape = res3.get_shape().as_list()
    fea = tf.reshape(res3,[feashape[0],feashape[2],feashape[3]],name = 'fea_rs')
    return fea


def rnn_layers(x,seq_length, hidden_num=100,layer_num = 3,class_n = 5):
    cells_fw = list()
    cells_bw = list()
    for i in range(layer_num):
        cell_fw = LSTMCell(hidden_num)
    cell_bw = LSTMCell(hidden_num)
    cells_fw.append(cell_fw)
    cells_bw.append(cell_bw)
    with tf.variable_scope('BDLSTM_rnn') as scope:
        lasth,_,_=stack_bidirectional_dynamic_rnn(cells_fw = cells_fw,cells_bw=cells_bw,\
                                                inputs = x,sequence_length = seq_length,dtype = tf.float32,scope=scope)
    batch_size = lasth.get_shape().as_list()[0]
    max_time = lasth.get_shape().as_list()[1]
    with tf.variable_scope('rnn_fnn_layer'):
        weight_out = tf.Variable(tf.truncated_normal([2,hidden_num],stddev=np.sqrt(2.0 / (2*hidden_num))),name='weights')
        biases_out = tf.Variable(tf.zeros([hidden_num]),name = 'bias')
        weight_class = tf.Variable(tf.truncated_normal([hidden_num,class_n],stddev=np.sqrt(2.0 / hidden_num)),name = 'weights_class')
        bias_class = tf.Variable(tf.zeros([class_n]),name = 'bias_class')
        lasth_rs = tf.reshape(lasth,[batch_size,max_time,2,hidden_num],name = 'lasth_rs')
        lasth_output = tf.nn.bias_add(tf.reduce_sum(tf.multiply(lasth_rs,weight_out),axis = 2),biases_out,name = 'lasth_bias_add')
        lasth_output_rs = tf.reshape(lasth_output,[batch_size*max_time,hidden_num],name = 'lasto_rs')
        logits = tf.reshape(tf.nn.bias_add(tf.matmul(lasth_output_rs,weight_class),bias_class),[batch_size,max_time,class_n],name = "rnn_logits_rs")
    return logits


def inference(x, seq_length):
    cnn_feature = getcnnfeature(x)
    feashape = cnn_feature.get_shape().as_list()
    ratio = sequence_len/feashape[1]
    logits = rnn_layers(cnn_feature,seq_length/ratio, class_n = 5)
    return logits,ratio


def loss(logits,seq_len,label):
    loss = tf.reduce_mean(tf.nn.ctc_loss(label,logits,seq_len,ctc_merge_repeated = True,time_major = False))
    """Note here ctc_loss will perform softmax, so no need to softmax the logits."""
    tf.summary.scalar('loss',loss)
    return loss


def train_step(loss):
    opt = tf.train.AdamOptimizer().minimize(loss)
    return opt


def prediction(logits,seq_length,label,top_paths=1):
    """
    Args:
        logits:Input logits from a RNN.Shape = [batch_size,max_time,class_num]
        seq_length:sequence length of logits. Shape = [batch_size]
        label:Sparse tensor of label.
        top_paths:The number of top score path to choice from the decorder.
    """
    logits = tf.transpose(logits,perm = [1,0,2])
    predict = tf.nn.ctc_beam_search_decoder(logits,seq_length,merge_repeated = False,top_paths = top_paths)[0]
    edit_d = list()
    for i in range(top_paths):
        tmp_d = tf.edit_distance(tf.to_int32(predict[i]), label, normalize=True)
        edit_d.append(tmp_d)
    tf.stack(edit_d,axis=0)
    d_min = tf.reduce_min(edit_d,axis=0)
    error = tf.reduce_mean(d_min,axis = 0)
    tf.summary.scalar('Error_rate',error)
    return error


def main():
    x = tf.placeholder(tf.float32,shape = [batch_size, sequence_len])
    seq_length = tf.placeholder(tf.int32, shape = [batch_size])
    y = tf.sparse.placeholder(tf.int32, shape=[batch_size, sequence_len])
    logits, ratio = inference(x, seq_length)
    ctc_loss = loss(logits,seq_length,y)
    opt = train_step(ctc_loss)
    error = prediction(logits,seq_length,y)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    summary = tf.summary.merge_all()
    sess = tf.Session(config = tf.ConfigProto(allow_soft_placement=True))
    dataset = Dataset(sys.argv[1])
    train, test = dataset.train_test_split()
    signal_seq = ExampleSequence(dataset, train, name='train', batch_size=batch_size)
    test_seq = ExampleSequence(dataset, test, name='test', batch_size=batch_size)
    val_batch = 0
    for i in range(len(signal_seq)):
        example = test_seq[i]
        feed_dict =  {x:example['the_input'],seq_length:example['input_length'],y:get_sparse(example)}
        loss_val,_ = sess.run([ctc_loss,opt],feed_dict = feed_dict)
        if i%10 ==0:
            example = test_seq[val_batch]
            val_batch = val_batch + 1
            feed_dict = {x:example['the_input'],seq_length:example['input_length'],y:get_sparse(example)}
            error_val = sess.run(error,feed_dict = feed_dict)
            print("Epoch %d, batch number %d, loss: %5.3f edit_distance: %5.3f"%(0,i,loss_val,error_val))