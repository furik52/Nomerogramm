import os
from pathlib import Path
from tensorflow import keras
import nomerogramm
import recognition
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
MODEL_PATH = Path('emnist_letters.h5').resolve()

model = keras.models.load_model('emnist_letters.h5')

def img_to_str(model: keras.models, image_file: str):
    result = ""
    for i in range(len(nomerogramm.letters)):
        dn = nomerogramm.letters[i+1][0] - nomerogramm.letters[i][0] - nomerogramm.letters[i][1] if i < len(nomerogramm.letters) - 1 else 0
        result += recognition.emnist_predict_img(model, nomerogramm.letters[i][2])
        if (dn > nomerogramm.letters[i][1]/4):
            result += ' '
    return result

model = keras.models.load_model(MODEL_PATH)