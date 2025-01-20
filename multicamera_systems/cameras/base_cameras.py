import cv2
import time


class GenericCamera:
    def __init__(self):
        pass

    def start_acquisition(self):
        pass

    def get_image(self):
        pass


class WebcamCamera(GenericCamera):
    def __init__(self, cam_id=0):
        self.cam = cv2.VideoCapture(cam_id)
        self.__image = None
        self.frame_counter = 0

    def get_image(self):
        try:
            ret, frame = self.cam.read()
            ts = 1000000 * time.time()
            n_frame = self.frame_counter
            self.frame_counter += 1
        except:
            ret = False
        if not ret:
            return (None, None, None)
        return float(n_frame), float(ts), frame