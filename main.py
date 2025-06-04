import customtkinter as ctk
from tkinter import filedialog
import cv2
import torch
import re
import numpy as np
import threading
from colour_detection.detect_color import detect_color
from lpr_net.model.lpr_net import build_lprnet
from lpr_net.rec_plate import rec_plate, CHARS
from settings import DEVICE
from object_detection.detect_car_YOLO import ObjectDetection
from track_logic import *
import settings


def get_frames(video_src: str):
    cap = cv2.VideoCapture(video_src)
    last_frame = None
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            last_frame = frame
            yield frame
        else:
            if last_frame is not None:
                while True:
                    yield last_frame
            else:
                break
    cap.release()


def preprocess(image: np.ndarray, size: tuple) -> np.ndarray:
    return cv2.resize(image, size, fx=0, fy=0, interpolation=cv2.INTER_CUBIC)


def get_boxes(results, frame):
    labels, cord = results
    x_shape, y_shape = frame.shape[1], frame.shape[0]

    labls_cords = {"numbers": [], "cars": [], "trucks": [], "busses": []}

    for i in range(len(labels)):
        row = cord[i]
        x1, y1, x2, y2 = (
            int(row[0] * x_shape),
            int(row[1] * y_shape),
            int(row[2] * x_shape),
            int(row[3] * y_shape),
        )

        if labels[i] == 0:
            labls_cords["numbers"].append((x1, y1, x2, y2))
        elif labels[i] == 1:
            labls_cords["cars"].append((x1, y1, x2, y2))
        elif labels[i] == 2:
            labls_cords["trucks"].append((x1, y1, x2, y2))
        elif labels[i] == 3:
            labls_cords["busses"].append((x1, y1, x2, y2))

    return labls_cords


def plot_boxes(cars_list, frame):
    for car in cars_list:
        car_type = car[2]
        x1_number, y1_number, x2_number, y2_number = car[0][0]
        number = car[0][1]
        x1_car, y1_car, x2_car, y2_car = car[1][0]
        colour = car[1][1]

        car_bgr = (0, 0, 255) if car_type == "car" else (0, 255, 0) if car_type == "truck" else (255, 0, 0)
        number_bgr = (255, 255, 255)

        cv2.rectangle(frame, (x1_car, y1_car), (x2_car, y2_car), car_bgr, 2)
        cv2.putText(frame, car_type + " " + colour, (x1_car, y2_car + 15), 0, 1, car_bgr, 2, cv2.LINE_AA)

        cv2.rectangle(frame, (x1_number, y1_number), (x2_number, y2_number), number_bgr, 2)
        cv2.putText(frame, number, (x1_number - 20, y2_number + 30), 0, 1, number_bgr, 2, cv2.LINE_AA)

    cv2.rectangle(frame, settings.DETECTION_AREA[0], settings.DETECTION_AREA[1], (0, 0, 0), 2)
    return frame


def check_roi(coords):
    detection_area = settings.DETECTION_AREA
    xc = int((coords[0] + coords[2]) / 2)
    yc = int((coords[1] + coords[3]) / 2)
    return detection_area[0][0] < xc < detection_area[1][0] and detection_area[0][1] < yc < detection_area[1][1]


def process_frame(raw_frame, detector, lprnet):
    proc_frame = preprocess(raw_frame, (640, 480))
    results = detector.score_frame(proc_frame)
    labls_cords = get_boxes(results, raw_frame)
    new_cars = check_numbers_overlaps(labls_cords)

    cars = []

    for car in new_cars:
        plate_coords = car[0]
        car_coords = car[1]

        if check_roi(plate_coords):
            x1_car, y1_car, x2_car, y2_car = car_coords
            car_box_image = raw_frame[y1_car:y2_car, x1_car:x2_car]
            colour = detect_color(car_box_image)
            car[1] = [car_coords, colour]

            x1_plate, y1_plate, x2_plate, y2_plate = plate_coords
            plate_box_image = raw_frame[y1_plate:y2_plate, x1_plate:x2_plate]
            plate_text = rec_plate(lprnet, plate_box_image)

            if re.match(r"[A-Z]{1}[0-9]{3}[A-Z]{2}[0-9]{2,3}", plate_text):
                car[0] = [plate_coords, plate_text + "_OK"]
            else:
                car[0] = [plate_coords, plate_text + "_NOK"]

            cars.append(car)

    drawn_frame = plot_boxes(cars, raw_frame)
    return preprocess(drawn_frame, settings.FINAL_FRAME_RES)


def process_file(filepath):
    is_video = filepath.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
    detector = ObjectDetection(settings.YOLO_MODEL_PATH, conf=settings.YOLO_CONF, iou=settings.YOLO_IOU, device=DEVICE)
    lprnet = build_lprnet(lpr_max_len=settings.LPR_MAX_LEN, phase=False, class_num=len(CHARS), dropout_rate=settings.LPR_DROPOUT)
    lprnet.to(DEVICE)
    lprnet.load_state_dict(torch.load(settings.LPR_MODEL_PATH, map_location=DEVICE))

    window_name = "Detection"

    if is_video:
        for raw_frame in get_frames(filepath):
            processed = process_frame(raw_frame, detector, lprnet)
            cv2.imshow(window_name, processed)

            key = cv2.waitKey(30) & 0xFF
            if key in [ord("q"), 27]:  # q or Esc
                break

            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
    else:
        raw_frame = cv2.imread(filepath)
        if raw_frame is None:
            print("Не удалось загрузить изображение.")
            return
        processed = process_frame(raw_frame, detector, lprnet)
        cv2.imshow(window_name, processed)
        cv2.waitKey(0)

    cv2.destroyAllWindows()



def open_file_dialog():
    filepath = filedialog.askopenfilename(
        title="Выберите видео или изображение",
        filetypes=[("Видео и изображения", "*.mp4 *.avi *.mov *.mkv *.jpg *.jpeg *.png *.bmp")]
    )
    if filepath:
        process_file(filepath)


def run_gui():
    ctk.set_appearance_mode("dark")  # dark mode
    ctk.set_default_color_theme("green")  # green accents

    app = ctk.CTk()
    app.geometry("400x280")
    app.title("Распознавание транспорта")

    label = ctk.CTkLabel(app, text="Выбор видео или фото", font=("Arial", 20))
    label.pack(pady=20)

    status_label = ctk.CTkLabel(app, text="", font=("Arial", 14))
    status_label.pack(pady=5)

    def open_file_dialog():
        filepath = filedialog.askopenfilename(
            title="Выберите видео или изображение",
            filetypes=[("Видео и изображения", "*.mp4 *.avi *.mov *.mkv *.jpg *.jpeg *.png *.bmp")]
        )
        if filepath:
            def process():
                status_label.configure(text="Идет обработка и открытие файла...")
                try:
                    process_file(filepath)
                finally:
                    status_label.configure(text="")

            threading.Thread(target=process, daemon=True).start()

    select_button = ctk.CTkButton(
        app,
        text="Выбрать файл",
        command=open_file_dialog,
        font=("Arial", 16),
        height=50,
        width=200,
        fg_color="#70FF9F",  # светло-зеленая кнопка
        text_color="black",
        hover_color="#55D88C"
    )
    select_button.pack(pady=10)

    app.mainloop()


if __name__ == "__main__":
    run_gui()
