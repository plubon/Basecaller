import tensorflow as tf
from config import ConfigReader
from data import DatasetExtractor
from model import ModelFactory


def train(config_path, dataset_path, output_path):
    config = ConfigReader(config_path).read()
    dataset_extractor = DatasetExtractor(dataset_path, config)
    dataset_train = dataset_extractor.extract_train()
    iterator = dataset_train.make_one_shot_iterator()
    signal, label, signal_len, _ = iterator.get_next()
    one_hot = tf.one_hot(tf.sparse.to_dense(label), 5, dtype=tf.int32)
    zero = tf.constant(0, dtype=tf.int32)
    where = tf.not_equal(one_hot, zero)
    indices = tf.where(where)
    values = tf.gather_nd(one_hot, indices)
    sparse_labels = tf.SparseTensor(indices, values, tf.shape(one_hot, out_type=tf.int64))
    model = ModelFactory.get(config.model_name, signal)
    signal_len_placeholder = tf.placeholder(tf.int32, shape=(None))
    label_placeholder = tf.sparse.placeholder(tf.int32)
    loss = tf.reduce_sum(tf.nn.ctc_loss(sparse_labels, model.logits, tf.cast(signal_len,
                                                                             dtype=tf.int32), time_major=False))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    decoded = tf.nn.ctc_beam_search_decoder(model.logits, tf.cast(signal_len, tf.int32), merge_repeated=False)[0][0]
    distance = tf.edit_distance(decoded, label)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        feed = {
            model.input: signal,
            signal_len_placeholder: signal_len,
            label_placeholder: sparse_labels
        }
        (loss_value, _, distance_value) = sess.run([loss, optimizer, distance])
        print(f'Loss:{loss_value}')
        print(f'Distance:{distance_value}')


if __name__ == "__main__":
    train("/home/piotr/Uczelnia/PracaMagisterska/Basecaller/basecaller/configs/test.json",
          '/home/piotr/Uczelnia/PracaMagisterska/Dane/dataset_chiron', None)
