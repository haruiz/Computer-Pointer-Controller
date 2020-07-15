import math
import os

import cv2
import mouse
import numpy as np
from face_detector import FaceDetector
from gaze_estimator import GazeEstimator
from head_pose_estimator import HeadPoseEstimator
from landmarks_detector import LandmarksDetector
from pipeline import PipelineOpt
from util import Utilities


class FeedData(PipelineOpt):
    def __init__(self, input_type: str, input_file: str = None):
        super(FeedData, self).__init__()
        self.input_type = input_type
        if input_type == 'video' or input_type == 'image':
            self.input_file = input_file
        # create video capture
        if self.input_type == 'video':
            self.cap = cv2.VideoCapture(self.input_file)
        elif self.input_type == 'cam':
            self.cap = cv2.VideoCapture(0)
        else:
            self.cap = cv2.imread(self.input_file)

    def generator(self):
        while self.has_next():
            if self.input_type != "image":
                ret, image = self.cap.read()
            else:
                ret, image = True, self.cap
            if not ret:
                # no frames has been grabbed
                break
            data = {"image": image}
            if self.filter(image):
                yield self.map(data)

    def map(self, data):
        """
        Transform the current image in the pipeline
        :rtype: object
        """
        if self.input_type == "cam":
            image = data["image"]
            data["image"] = cv2.flip(image, 1)

        return data

    def cleanup(self):
        """Closes video file or capturing device.
        This function should be triggered after the pipeline completes.
        """
        if not self.input_type == 'image':
            self.cap.release()


class DetectFaces(PipelineOpt):
    def __init__(self, precision="FP32-INT1", device="CPU"):
        super(DetectFaces, self).__init__()
        # initialize the detector model
        self.model = FaceDetector("face-detection-adas-binary-0001", precision, device)
        self.model.load_model()

    def map(self, data):
        """
        Transform face location relative coordinates to absolute coordinates
        :rtype: object
        """
        # grab the current generated image
        image = data["image"]
        data["face"] = None
        bounding_boxes = self.model.predict(image)
        # check if any face was detected
        if len(bounding_boxes) > 0:
            height, width, _ = np.shape(image)
            # transform bounding boxes
            x_min, y_min, x_max, y_max = bounding_boxes[0]
            x_min = math.floor(x_min * width)
            y_min = math.floor(y_min * height)
            x_max = math.floor(x_max * width)
            y_max = math.floor(y_max * height)
            # save the face location in the data dictionary,
            # so it can be recovered from other operation in the pipeline
            data["face"] = (x_min, y_min, x_max, y_max)
        return data


class DetectLandmarks(PipelineOpt):
    def __init__(self, precision="FP16", device="CPU"):
        super(DetectLandmarks, self).__init__()
        # load the model
        self.model = LandmarksDetector("landmarks-regression-retail-0009", precision, device)
        self.model.load_model()

    def map(self, data):
        image = data["image"]  # grabbing current image in the pipeline
        face_roi = data["face"]  # grabbing the face rois detected in the previous step
        data["landmarks"] = []
        # check if any face was detected
        if face_roi:
            x_min, y_min, x_max, y_max = face_roi
            # extract the face roi from the source image
            face_roi_numpy = image[y_min:y_max, x_min: x_max]
            # detect landmarks in cropped region
            face_landmarks_points = self.model.predict(face_roi_numpy)
            height, width, _ = np.shape(face_roi_numpy)
            # transform the point locations based on the original image
            for (x, y) in face_landmarks_points:
                x = math.floor(x_min + (x * width))
                y = math.floor(y_min + (y * height))
                # save the landmarks points into the data dictionary, so
                # they can be accessed from another task in the pipeline
                data["landmarks"].append([x, y])
        return data


class EstimateHeadPose(PipelineOpt):
    def __init__(self, precision="FP16", device="CPU"):
        super(EstimateHeadPose, self).__init__()
        # load the model
        self.model = HeadPoseEstimator("head-pose-estimation-adas-0001", precision, device)
        self.model.load_model()

    def map(self, data):
        image = data["image"]  # grabbing current image in the pipeline
        face_roi = data["face"]  # grabbing the face rois detected in the previous step
        data["head_pose"] = None
        if face_roi:
            x_min, y_min, x_max, y_max = face_roi
            # extract the face roi from the source image
            face_roi_numpy = image[y_min:y_max, x_min: x_max]
            # estimate head pose base on the cropped region
            head_yaw, head_pitch, head_roll = self.model.predict(face_roi_numpy)
            data["head_pose"] = (head_yaw, head_pitch, head_roll)
        return data


class EstimateGaze(PipelineOpt):
    def __init__(self, precision):
        super(EstimateGaze, self).__init__()
        self.model = GazeEstimator("gaze-estimation-adas-0002", precision)
        self.model.load_model()

    def map(self, data):
        image = data["image"]
        landmarks = data["landmarks"]
        head_pose = data["head_pose"]
        if landmarks and head_pose:
            # extract left and right eye
            left_eye_x, left_eye_y = landmarks[0]  # right_eye_coords
            right_eye_x, right_eye_y = landmarks[1]  # left_eye_coords
            gaze = self.model.predict(image, left_eye_x, left_eye_y, right_eye_x, right_eye_y, head_pose, 30)
            height, width = image.shape[:2]
            arrow_length = 0.9 * height
            gaze_arrow_x = gaze[0] * arrow_length
            gaze_arrow_y = -gaze[1] * arrow_length
            data["gaze"] = (left_eye_x, left_eye_y, right_eye_x, right_eye_y, gaze_arrow_x, gaze_arrow_y)
        return data


class VisualizeResults(PipelineOpt):
    def __init__(self, window_name=None, x=0, y=0, debug=True):
        super(VisualizeResults, self).__init__()
        self.debug = debug
        self.window_name = window_name
        cv2.startWindowThread()
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow(self.window_name, x, y)

    def map(self, data):
        image = np.copy(data["image"])  # grabbing current image in the pipeline

        if "gaze" in data and data["gaze"]:
            left_eye_x, left_eye_y, right_eye_x, right_eye_y, gaze_arrow_x, gaze_arrow_y = data["gaze"]
            x_min, y_min, x_max, y_max = data["face"]
            landmarks = data["landmarks"]
            if self.debug:
                # draw detected face
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 55, 255), 1)

                # draw landmark points
                for index, (x, y) in enumerate(landmarks):
                    cv2.rectangle(image, (x, y), (x + 10, y + 10), (255, 0, 255), -1)
                    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
                    cv2.putText(image, f"{index}", (x - 10, y - 10), font, 1, (255, 0, 255))

                # draw gaze
                cv2.arrowedLine(image, (left_eye_x, left_eye_y),
                                (int(left_eye_x + gaze_arrow_x), int(left_eye_y + gaze_arrow_y)), (184, 113, 57), 2)
                cv2.arrowedLine(image, (right_eye_x, right_eye_y),
                                (int(right_eye_x + gaze_arrow_x), int(right_eye_y + gaze_arrow_y)), (184, 113, 57), 2)

            # move the mouse
            mouse.move(int(left_eye_x + gaze_arrow_x), int(left_eye_y + gaze_arrow_y), absolute=True, duration=0.2)

        cv2.imshow(self.window_name, image)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            raise StopIteration
        return data

    def cleanup(self):
        cv2.destroyWindow(self.window_name)  # destroy window


if __name__ == '__main__':
    os.environ["MODELS_PATH"] = "./models/intel"  # setup the folder were the models will be downloaded
    # feed_data = FeedData(input_type="image", input_file="./bin/image.jpg")
    feed_data = FeedData(input_type="video", input_file="./bin/demo.mp4")
    # feed_data = FeedData(input_type="cam")
    detect_faces = DetectFaces("FP32-INT1")
    detect_landmarks = DetectLandmarks("FP16")
    estimate_head_pose = EstimateHeadPose("FP16")
    estimate_gaze = EstimateGaze("FP16")
    visualize_result = VisualizeResults("VIDEO", debug=True)
    pipeline = (
            feed_data | detect_faces | detect_landmarks | estimate_head_pose | estimate_gaze | visualize_result
    )

    try:
        # run pipeline
        for _ in pipeline:
            pass
        # print stats
        models = [
            detect_faces.model,
            detect_landmarks.model,
            estimate_head_pose.model,
            estimate_gaze.model
        ]

        Utilities.make_prediction_time_chart(models)
        Utilities.make_model_loading_time_chart(models)
        Utilities.make_model_perform_chart_with_annot(models)

    except StopIteration:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        feed_data.cleanup()
        visualize_result.cleanup()
