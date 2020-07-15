from model import OpenVINOModel
import numpy as np
from util import Utilities


class GazeEstimator(OpenVINOModel):
    def __init__(self, model_name, precision, device='CPU', extensions=None):
        super(GazeEstimator, self).__init__(model_name, precision, device, extensions)

    def preprocess_input(self, image, left_eye_x,left_eye_y, right_eye_x,right_eye_y, head_pose, offset=30) -> dict:
        left_eye_numpy = image[left_eye_y - offset:left_eye_y + offset, left_eye_x - offset:left_eye_x + offset]
        right_eye_numpy = image[right_eye_y - offset:right_eye_y + offset, right_eye_x - offset:right_eye_x + offset]
        # batch inputs
        left_eye_numpy = Utilities.batch_image(left_eye_numpy, 60, 60)
        right_eye_numpy = Utilities.batch_image(right_eye_numpy, 60, 60)
        head_yaw, head_pitch, head_roll = head_pose
        head_pose_angles = np.expand_dims([head_yaw, head_pitch, head_roll], 0)
        return {
                "left_eye_image": left_eye_numpy,
                "right_eye_image": right_eye_numpy,
                "head_pose_angles": head_pose_angles
        }

    def preprocess_output(self, outputs):
        return np.squeeze(outputs)
