import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

YOLO_MODEL_PATH = "object_detection/YOLOS_cars.pt"
LPR_MODEL_PATH = "lpr_net/model/weights/LPRNet__iteration_2000_28.09.pth"

YOLO_CONF = 0.5
YOLO_IOU = 0.4
LPR_MAX_LEN = 9
LPR_DROPOUT = 0

FINAL_FRAME_RES = (640, 480)
DETECTION_AREA = [(0, 650), (1920, 1000)]
