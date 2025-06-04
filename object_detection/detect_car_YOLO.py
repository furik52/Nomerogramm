import torch

class ObjectDetection:


    def __init__(self, model_path, conf, iou, device):
        self.__model_path = model_path
        self.model = self.load_model()

        self.model.conf = conf
        self.model.iou = iou

        self.device = device

    def load_model(self):

        model = torch.hub.load("ultralytics/yolov5", "custom", path=self.__model_path)
        return model

    def score_frame(self, frame):

        self.model.to(self.device)
        results = self.model([frame])
        labels, cord = (
            results.xyxyn[0][:, -1].to("cpu").numpy(),
            results.xyxyn[0][:, :-1].to("cpu").numpy(),
        )
        return labels, cord

