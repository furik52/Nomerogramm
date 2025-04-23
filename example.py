from pathlib import Path
from tensorflow import keras
import nomerogramm
import recognition

def load_model(model_path: str = 'license_plate_model.h5') -> keras.Model:
    """Загрузка обученной модели"""
    return keras.models.load_model(model_path)

def recognize_number_plate(image_path: str) -> str:
    """Основная функция распознавания номера"""
    model = load_model()
    chars = nomerogramm.main(image_path)
    
    if not chars:
        return "Номер не обнаружен"
    
    return recognition.recognize_plate(model, chars)
