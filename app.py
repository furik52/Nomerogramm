from customtkinter import *

#dark/light mode + default objects color theme
set_appearance_mode('dark')
set_default_color_theme('green')

root = CTk()
root.title('Номерограмм')
root.geometry('800x600')

def click_handler():
    input_file = filedialog.askopenfile().name
    print('Путь до файла:', input_file)

btn = CTkButton(master=root, text='Выберите фотографию', corner_radius=16, command=click_handler)
btn.place(relx=0.5, rely=0.5, anchor='center')


root.mainloop()