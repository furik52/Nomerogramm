import tensorflow as tf
from neural_network import car_plate_model

NUM_CLASSES = 22

def parse_example(example_proto):
    features = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    }
    parsed = tf.io.parse_single_example(example_proto, features)

    # Декодируем изображение
    image = tf.io.decode_jpeg(parsed['image/encoded'], channels=1)
    image = tf.image.resize(image, [28, 28])
    image = tf.cast(image, tf.float32) / 255.0

    # Преобразуем label (используем первый элемент, уменьшаем на 1 и кастим в int32)
    label_sparse = parsed['image/object/class/label']
    label = tf.sparse.to_dense(label_sparse)[0] - 1
    label = tf.cast(label, tf.int32)

    return image, label

def tf_parse_example(example_proto):
    image, label = tf.py_function(parse_example, [example_proto], [tf.float32, tf.int32])
    image.set_shape([28, 28, 1])
    label.set_shape([])
    return image, label

def load_dataset(tfrecord_path):
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(tf_parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
    return dataset

# Пути
train_dataset = load_dataset("./train/SimbolPlate.tfrecord")
valid_dataset = load_dataset("./valid/SimbolPlate.tfrecord")

# Модель
model = car_plate_model()
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Обучение модели с использованием ранней остановки
model.fit(train_dataset, epochs=50, validation_data=valid_dataset, callbacks=[early_stopping])

# Сохраняем
model.save("license_plate_model.h5")
print("✅ Модель сохранена как license_plate_model.h5")
