from tensorflow import keras
import reading_base
import example
import os
from pathlib import Path


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

model_path = Path(os.getcwd()) / "emnist_letters.h5"

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

history = example.model.fit(
    x=reading_base.X_train,
    y=reading_base.x_train_cat,
    validation_data=(reading_base.X_test, reading_base.y_test_cat),
    callbacks=[learning_rate_reduction, checkpoint],
    batch_size=64,
    epochs=30,
    verbose=2
)

example.model.save(str(model_path))
print(f"Модель успешно сохранена: {model_path}")
