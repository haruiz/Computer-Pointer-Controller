from model import OpenVINOModel
import numpy as np

class HeadPoseEstimator(OpenVINOModel):
    def __init__(self, model_name, precision, device='CPU', extensions=None):
        super(HeadPoseEstimator, self).__init__(model_name, precision, device, extensions)

    def preprocess_output(self, outputs):
        """
        Each output contains one float value that represents value in Tait-Bryan angles (yaw, pitch or roll).
        :rtype: object
        """
        yaw = np.squeeze(outputs["angle_y_fc"])
        pitch = np.squeeze(outputs["angle_p_fc"])
        roll = np.squeeze(outputs["angle_r_fc"])
        return yaw, pitch, roll
