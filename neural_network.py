from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

# Новый список символов: 10 арабских цифр и 20 печатных русских букв (чаще всего встречающихся на номерах)
Car_plate_labels = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'А', 'В', 'Е', 'К', 'М', 'Н', 'О', 'Р', 'С', 'Т', 'У', 'Х',
    'P', 'A', 'E', 'Y', 'B', 'C', 'H', 'K'  ]

# Функция для создания модели под CAr_plate_OCR_Dataset
# Получает на вход изображение 28x28 в оттенках серого (1 канал)
def car_plate_model():
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='valid', input_shape=(28, 28, 1), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(Car_plate_labels), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
