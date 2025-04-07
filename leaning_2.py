import tensorflow_datasets as tfds
import os
import tensorflow as tf
import neural_network

allowed_labels = [*range(10), 10, 11, 12, 14, 17, 22, 24, 25, 30, 31, 33]

def preprocess(data):
    image = tf.cast(data['image'], tf.float32) / 255.0
    label = tf.one_hot(data['label'], depth=75)
    return image, label

def filter_fn(example):
    return tf.reduce_any(example['label'] == allowed_labels)

ds = tfds.load('emnist/byclass', split='train[:100%]', data_dir=os.getcwd(),shuffle_files=True).filter(filter_fn)

ds = ds.map(preprocess).batch(32)

model = neural_network.emnist_model()
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit(ds, epochs=5)
model.save('emnist_letters.h5', save_format='h5')
