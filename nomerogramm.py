import cv2
import numpy as np
import os
import random
from typing import List, Tuple, Any

# Функция для нахождения угла наклона изображения
def find_skew_angle(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Применим бинаризацию для выделения контуров
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Находим контуры
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Находим прямоугольник, охватывающий контуры
    points = np.vstack([cnt.squeeze() for cnt in contours if len(cnt) > 1])  # Строим набор точек
    rect = cv2.minAreaRect(points)  # Получаем минимальный охватывающий прямоугольник
    angle = rect[2]
    # Корректируем угол наклона
    if angle < -45:
        angle = 90 + angle
    return angle

# Функция для поворота изображения на заданный угол
def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    # Получаем центр изображения
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Строим матрицу для поворота
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Поворачиваем изображение
    rotated = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

# Функция для аугментации изображения
def augment_image(image: np.ndarray) -> np.ndarray:
    # Поворот изображения
    angle = random.uniform(-15, 15)  # Угол поворота от -15 до 15 градусов
    rotated = rotate_image(image, angle)

    # Масштабирование
    scale = random.uniform(0.8, 1.2)  # Масштабирование от 80% до 120%
    width = int(rotated.shape[1] * scale)
    height = int(rotated.shape[0] * scale)
    resized = cv2.resize(rotated, (width, height), interpolation=cv2.INTER_LINEAR)

    # Сдвиг изображения
    shift_x = random.randint(-10, 10)
    shift_y = random.randint(-10, 10)
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    shifted = cv2.warpAffine(resized, M, (resized.shape[1], resized.shape[0]))

    return shifted

def preprocess_image(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Коррекция наклона изображения
    angle = find_skew_angle(image)
    rotated = rotate_image(image, angle)
    
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    return thresh


def find_plate_contours(image: np.ndarray) -> List[Tuple[int, int, int, int]]:
    processed = preprocess_image(image)
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    plates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / h
        if 2.0 < aspect_ratio < 5.0 and w > 100 and h > 30:
            plates.append((x, y, w, h))
    return plates

def extract_characters(plate_roi: np.ndarray, out_size: int = 28) -> List[Tuple[int, int, np.ndarray]]:
    gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if not os.path.exists("debug_chars"):
        os.makedirs("debug_chars")

    chars = []
    count = 0
    for idx, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        if h > plate_roi.shape[0] * 0.4:
            char_img = thresh[y:y+h, x:x+w]

            size_max = max(w, h)
            square = 255 * np.ones((size_max, size_max), dtype=np.uint8)
            x_offset = (size_max - w) // 2
            y_offset = (size_max - h) // 2
            square[y_offset:y_offset + h, x_offset:x_offset + w] = char_img

            resized = cv2.resize(square, (out_size, out_size), interpolation=cv2.INTER_AREA)
            norm_img = resized.astype("float32") / 255.0
            norm_img = np.expand_dims(norm_img, axis=-1)  # (28, 28, 1)

            chars.append((x, w, norm_img))

            cv2.imwrite(f"debug_chars/char_{count}.png", resized)
            count += 1

    chars.sort(key=lambda c: c[0])
    return chars



def main(path_to_file: str, out_size: int = 28) -> List[Tuple[int, int, np.ndarray]]:
    image = cv2.imdecode(np.fromfile(path_to_file, dtype=np.uint8), cv2.IMREAD_COLOR)
    plates = find_plate_contours(image)
    
    if not plates:
        return []
    
    x, y, w, h = plates[0]
    plate_roi = image[y:y+h, x:x+w]
    return extract_characters(plate_roi, out_size)
