from pathlib import Path
from tensorflow import keras
import nomerogramm
import recognition
import os
from typing import Union

# Конфигурация среды
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
MODEL_PATH = Path('emnist_letters.h5').resolve()

def img_to_str(model: keras.Model, image_file: Union[str, Path]) -> str:

    # Получаем структурированные данные об изображении
    letters = nomerogramm.main(str(image_file))
    
    # Валидация результатов обработки изображения
    assert len(letters) > 0, "Не обнаружено символов на изображении"
    
    s_out = []
    for i in range(len(letters)):
        current_char = letters[i]
        
        # Извлекаем параметры символа
        x_pos, width, char_img = current_char[0], current_char[1], current_char[2]
        
        # Определяем расстояние до следующего символа
        next_x = letters[i+1][0] if i < len(letters)-1 else x_pos + width
        distance = next_x - (x_pos + width)
        
        # Распознаем символ
        predicted_char = recognition.emnist_predict_img(model, char_img)
        s_out.append(predicted_char)
        
        # Добавляем пробел при значительном расстоянии
        if distance > width/4:
            s_out.append(' ')
    
    return ''.join(s_out)

# Загрузка модели с проверками
assert MODEL_PATH.exists(), f"Файл модели {MODEL_PATH} не найден"
model = keras.models.load_model(MODEL_PATH)
assert isinstance(model, keras.Model), "Некорректный тип загруженной модели"

#s_out = img_to_str(model, "hello_world.png")
#print(s_out)