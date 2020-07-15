import math
import more_itertools
from model import OpenVINOModel
import numpy as np
import cv2


class LandmarksDetector(OpenVINOModel):
    def __init__(self, model_name, precision, device='CPU', extensions=None):
        super(LandmarksDetector, self).__init__(model_name, precision, device, extensions)


    def preprocess_output(self, outputs):
        """
         The net outputs a blob with the shape: [1, 10],
         containing a row-vector of 10 floating point values for five landmarks coordinates in the form (x0, y0, x1, y1, ..., x5, y5).
         All the coordinates are normalized to be in range [0,1].
        :param outputs:
        :return: list of points that conform the face landmarks
        """
        points = np.squeeze(outputs)
        return list(more_itertools.chunked(points, n=2))

