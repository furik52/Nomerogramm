import tensorflow_datasets as tfds
import os
import tensorflow as tf

allowed_labels = [*range(10), 10, 11, 12, 14, 17, 22, 24, 25, 30, 31, 33 ]

def filter_fn(example):
    return tf.reduce_any(example['label'] == allowed_labels)

ds = tfds.load('emnist/byclass', split='train[::5]', data_dir=os.getcwd(),shuffle_files=True).filter(filter_fn)

assert isinstance(ds, tf.data.Dataset)
print(ds)
