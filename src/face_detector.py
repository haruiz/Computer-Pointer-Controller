import math

from model import OpenVINOModel
import numpy as np
import cv2


class FaceDetector(OpenVINOModel):
    def __init__(self, model_name, precision, device='CPU', extensions=None):
        super(FaceDetector, self).__init__(model_name, precision, device, extensions)

    def preprocess_output(self, outputs: np.ndarray) -> list:
        """
        process the face detection model output and create a list of
        boxes with the coordinates where the faces wer found
        :param outputs: model outputs
        :return: list with the bounding boxes
        """
        coords = []
        outputs = np.squeeze(outputs)
        for box in outputs:
            _, _, conf, x_min, y_min, x_max, y_max = box
            if conf > self.threshold:
                coords.append([x_min, y_min, x_max, y_max])
        return coords

