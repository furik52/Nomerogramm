import tensorflow as tf
from datasets import load_dataset
from PIL import Image
import numpy as np
import neural_network

# Загрузка датасета
dataset = load_dataset("AY000554/Car_plate_OCR_dataset", cache_dir="car_plate_dataset")
train_data = dataset["train"]
print(train_data[0])


# Получаем уникальные символы
all_labels = sorted(list(set(example["label"] for example in train_data)))
label_to_index = {label: idx for idx, label in enumerate(all_labels)}
num_classes = len(all_labels)

def preprocess(example):
    # Преобразуем изображение к 28x28 grayscale, numpy → Tensor
    image = example["image"].convert("L").resize((28, 28))
    image = np.array(image).astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=-1)  # (28, 28, 1)
    
    label = label_to_index[example["label"]]
    label = tf.one_hot(label, num_classes)
    
    return image, label

# Преобразуем весь датасет
images = []
labels = []

for example in train_data:
    image, label = preprocess(example)
    images.append(image)
    labels.append(label)

images = np.stack(images)
labels = np.stack(labels)

# Обучение
model = neural_network.build_model(num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(images, labels, batch_size=64, epochs=10)
model.save("carplate_model.h5")
