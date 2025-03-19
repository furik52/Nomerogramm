import keras
import nomerogramm
import recognition
from typing import *

def img_to_str(model: Any, image_file: str):
    letters = nomerogramm.main(image_file)
    s_out = ""
    for i in range(len(letters)):
        dn = letters[i+1][0] - letters[i][0] - letters[i][1] if i < len(letters) - 1 else 0
        s_out += recognition.emnist_predict_img(model, letters[i][2])
        if (dn > letters[i][1]/4):
            s_out += ' '
    return s_out
model = keras.models.load_model('emnist_letters.h5')
s_out = img_to_str(model, "hello_world.png")
print(s_out)