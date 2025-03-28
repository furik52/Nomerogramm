from tensorflow import keras
import reading_base
import example
import os
from pathlib import Path

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 1. Проверка модели
assert hasattr(example, 'model'), "Модель не найдена в модуле example"
assert example.model is not None, "Модель не инициализирована"

# 2. Настройка путей
model_path = Path(os.getcwd()) / "emnist_letters.h5"

# 3. Коллбэки
learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(
    monitor='val_accuracy',
    patience=3,
    verbose=1,
    factor=0.5,
    min_lr=0.00001
)

checkpoint = keras.callbacks.ModelCheckpoint(
    filepath=model_path,
    monitor='val_accuracy',
    save_best_only=True
)

# 4. Обучение
history = example.model.fit(
    x=reading_base.X_train,
    y=reading_base.x_train_cat,
    validation_data=(reading_base.X_test, reading_base.y_test_cat),
    callbacks=[learning_rate_reduction, checkpoint],
    batch_size=64,
    epochs=30,
    verbose=2
)

# 5. Сохранение
example.model.save(str(model_path))
print(f"Модель успешно сохранена: {model_path}")
