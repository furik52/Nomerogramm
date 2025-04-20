from datasets import load_dataset
from PIL import Image
import numpy as np
import tensorflow as tf
import neural_network  # твоя модель

# Загружаем посимвольный датасет
dataset = load_dataset("AY000554/Car_plate_OCR_characters_dataset", cache_dir="car_plate_dataset")
train_data = dataset["train"]
