import keras
import reading_base
import example

learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(
    monitor='val_accuracy', 
    patience=3, 
    verbose=1, 
    factor=0.5, 
    min_lr=0.00001)

example.model.fit(reading_base.X_train, reading_base.x_train_cat, validation_data=(reading_base.X_test, reading_base.y_test_cat), callbacks=[learning_rate_reduction], batch_size=64, epochs=30)

example.model.save('emnist_letters.h5')