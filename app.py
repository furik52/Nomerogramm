from customtkinter import *
import cv2
import numpy as np
import nomerogramm
from tensorflow import keras
import example

#dark/light mode + default objects color theme
set_appearance_mode('dark')
set_default_color_theme('green')

#os.system('cls')
root = CTk()
root.title('Номерограмм')
root.geometry('800x600')
def click_handler():
    result  = example.img_to_str(keras.models.load_model('emnist_letters.h5'), filedialog.askopenfile().name)
    print('результат', result)

btn = CTkButton(master=root, text='Выберите фотографию', corner_radius=16, command=click_handler)
btn.place(relx=0.5, rely=0.5, anchor='center')

root.lift()
root.attributes('-topmost',True)
root.after_idle(root.attributes,'-topmost',False)
root.mainloop()