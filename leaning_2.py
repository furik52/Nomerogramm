import tensorflow_datasets as tfds
import tensorflow as tf
ds = tfds.load('emnist', split='train', shuffle_files=True)
assert isinstance(ds, tf.data.Dataset)
print(ds)
