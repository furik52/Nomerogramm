import torch
import os
from Ipr_net.object_detection.detect_car_YOLO import LPRNet  # Импортируйте ваш класс модели

def test_model_loading():
    # 1. Инициализация модели
    model = LPRNet()  # Используйте правильный класс вашей архитектуры
    
    # 2. Загрузка весов
    checkpoint_path = r'C:\Users\vladf\...\LPRNet__iteration_2000_28.09.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 3. Проверка структуры файла
    if isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint['state_dict'])  # Если есть дополнительные данные
    else:
        model.load_state_dict(checkpoint)  # Если файл содержит только state_dict
    
    # 4. Тестирование
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
        print(f"✅ Успешно! Размер выхода: {output.shape}")

if __name__ == "__main__":
    test_model_loading()
