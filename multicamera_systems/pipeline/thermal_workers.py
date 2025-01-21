import PIL
from PIL import ImageDraw
from scipy.spatial import ConvexHull
from multicamera_systems.pipeline import MultimodalWorker
import numpy as np
from PyQt6.QtCore import pyqtSignal, QThread
import time
from multicamera_systems.tfake import ThermalLandmarks
from neurovc.util.IO_util import map_temp, normalize_color, draw_landmarks
import cv2
import mediapipe as mp
from collections import deque


class ThermalLandmarkWorker(MultimodalWorker):
    new_frame = pyqtSignal(np.ndarray, float, float)

    def __init__(self):
        MultimodalWorker.__init__(self)
        self.landmarker = ThermalLandmarks()

    def run(self):
        ts_baseline = -1

        while True:
            for (i, cam) in enumerate(self._cams):
                if len(cam) < 1:
                    continue
                (n_frame, ts, frame) = cam[-1]
                if ts_baseline == -1:
                    ts_baseline = ts

                frame_raw = map_temp(frame, "A655")
                frame = normalize_color(frame, color_map=cv2.COLORMAP_INFERNO)

                landmarks = self.landmarker.process(frame_raw)[0]
                frame = draw_landmarks(frame, landmarks)

                self.new_frame.emit(frame, 0, ts - ts_baseline)

            time.sleep(1/30)


class TemperatureWorker(MultimodalWorker):
    new_frame = pyqtSignal(np.ndarray, float, float)
    change_eye_temp = pyqtSignal(float, float)
    change_mouth_temp = pyqtSignal(float, float)
    change_ear_signal = pyqtSignal(float, float)

    def _wrap_cam(self, cam):
        frame_deque = deque(maxlen=10)
        cam.new_frame.connect(
            lambda frame, n_frame, ts: frame_deque.append((n_frame, ts, frame)))
        return frame_deque

    def __init__(self):
        QThread.__init__(self)
        MultimodalWorker.__init__(self)
        self.respiration = []
        self.temperature = []
        self.ear = []
        self.landmarker = ThermalLandmarks()

    def get_temperature(self):
        return self.temperature

    def get_respiration(self):
        return self.respiration

    def get_ear(self):
        return self.ear

    def run(self):

        old_nframe = -1

        ts_baseline = -1

        ts_array = []
        temp_array = []
        resp_array = []
        ear_array = []


        ears_blinking_rate = deque(maxlen=350)
        ears_timestamps = deque(maxlen=350)
        while(True):
            # always waiting for the thermal frame and then getting the most recent ximea frames:
            if len(self.thermal_cam) < 1 \
                    or len(self.left_cam) < 1 or len(self.right_cam) < 1:
                continue

            (n_frame_thermal, ts_thermal, thermal) = self.thermal_cam[-1]
            if n_frame_thermal == old_nframe:
                continue

            if ts_baseline == -1:
                ts_baseline = ts_thermal

            old_nframe = n_frame_thermal

            (n_frame, ts, left) = self.left_cam[-1]
            (n_frame, ts, right) = self.right_cam[-1]

            thermal = map_temp(thermal[:480, :640], "A655")
            thermal_raw = thermal.copy()

            thermal_mapped = self.landmarker.process(thermal)[0]
            thermal = normalize_color(thermal, color_map=cv2.COLORMAP_INFERNO)
            if -1 in thermal_mapped:
                thermal_mapped = None

            if thermal_mapped is not None:

                #p1 = 27
                #p2 = 23

                #p3 = 257
                #p4 = 253

                temps = []
                mask = get_thermal_mask(thermal, pose_mapped[mouth_boundary])
                mouth = np.sum(thermal_raw[mask != 0]) / (0.000001 + np.sum(mask != 0))
                mask1 = get_thermal_mask(thermal, pose_mapped[eye_left_boundary])
                mask2 = get_thermal_mask(thermal, pose_mapped[eye_right_boundary])
                mask1[mask2 != 0] = 255
                mask1[mask2 != 0] = 255
                try:
                    eye = np.max(thermal_raw[mask1 != 0])
                except:
                    eye = 0

                mask1[mask != 0] = 255

                thermal_mask = thermal.copy()
                thermal_mask[np.repeat(np.expand_dims(mask1, 2), 3, axis=2) == 0] = 0
                # cv2.imshow("thermal mask", thermal_mask)

                self.change_eye_temp.emit(eye, ts_thermal - ts_baseline)
                self.change_mouth_temp.emit(mouth, ts_thermal - ts_baseline)


            #cv2.imshow("left", left)
            #cv2.imshow("right", right)

            if thermal_mapped is not None:
                #cv2.imshow("thermal", thermal_mapped)
                self.new_frame.emit(thermal_mapped, 0, ts_thermal - ts_baseline)
            else:
                #cv2.imshow("thermal", thermal)
                self.new_frame.emit(thermal, 0, ts_thermal - ts_baseline)
            #cv2.waitKey(1)
            time.sleep(1/300)


class ThermalVizThread(MultimodalWorker):
    new_frame = pyqtSignal(np.ndarray, float, float)
    change_plot_signal = pyqtSignal(float, float)
    change_ear_signal = pyqtSignal(float, float)
    def __init__(self):
        MultimodalWorker.__init__(self)

        self.temperature = []
        self.respiration = []
        # self.inpainting_func = Inpainter()

    def run(self):
        ts_baseline = -1
        ts_array = []
        temperature = []
        respiration = []

        while True:
            for (i, cam) in enumerate(self._cams):
                if len(cam) < 1:
                    continue
                # (n_frame, ts, frame) = cam.frame_deque.pop()
                (n_frame, ts, frame) = cam[-1]
                if ts_baseline == -1:
                    ts_baseline = ts

                n, m = frame.shape[0:2]

                frame_raw = map_temp(frame, "A655")

                self.change_plot_signal.emit(frame_raw.max(), ts - ts_baseline)

                frame = frame_raw
                frame[frame < 20] = 20
                frame[frame > 39] = 39
                # frame = normalize_color(frame, color_map=cv2.COLORMAP_INFERNO)
                frame = normalize_color(frame, color_map=cv2.COLORMAP_INFERNO)

                self.new_frame.emit(frame, 0, ts - ts_baseline)

            time.sleep(1/30)

    def get_temp(self):
        return self.temperature

    def get_respiration(self):
        return self.respiration


def get_thermal_mask(frame, points):
    img = PIL.Image.new('F', (frame.shape[:2][::-1]), 0.0)
    #  points = landmarks[idx]

    hull = ConvexHull(points)

    args = np.argsort(hull.simplices, axis=0)
    _, idx = np.unique(hull.simplices[args[:, 0]][:, 0], return_index=True)

    points = [(x, y) for x, y in points[hull.simplices[args[:, 0]][:, 0][idx]]]
    ImageDraw.Draw(img).polygon(points, outline=255, fill=255)
    img = np.array(img)

    return img
