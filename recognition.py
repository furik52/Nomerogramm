import keras
import neural_network
import numpy as np
#model = keras.models.load_model('emnist_letters.h5')

def emnist_predict_img(model, img):
    img_arr = np.expand_dims(img, axis=0)
    img_arr = 1 - img_arr/255.0
    img_arr[0] = np.rot90(img_arr[0], 3)
    img_arr[0] = np.fliplr(img_arr[0])
    img_arr = img_arr.reshape((1, 28, 28, 1))

    predict = model.predict([img_arr])
    result = np.argmax(predict, axis=1)
    return chr(neural_network.emnist_labels[result[0]])