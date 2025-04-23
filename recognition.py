import numpy as np
import keras
import neural_network

def preprocess_char(img: np.ndarray) -> np.ndarray:
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=-1)
    return np.expand_dims(img, axis=0)

def predict_char(model: keras.Model, img: np.ndarray) -> str:
    processed = preprocess_char(img)
    pred = model.predict(processed, verbose=0)
    idx = np.argmax(pred)
    print(f"[DEBUG] Предсказанный индекс: {idx}, класс: {neural_network.Car_plate_labels[idx]}")
    return neural_network.Car_plate_labels[idx]

def recognize_plate(model: keras.Model, chars: list) -> str:
    result = []
    for x, w, char_img in chars:
        char = predict_char(model, char_img)
        result.append(char)
        if len(result) in (3, 6):
            result.append(' ')
    return ''.join(result).strip()
