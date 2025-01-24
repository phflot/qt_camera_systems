import PIL
from PIL import ImageDraw
from scipy.spatial import ConvexHull
from multicamera_systems.pipeline import MultimodalWorker
import numpy as np
from PyQt6.QtCore import pyqtSignal, QThread, pyqtSlot, Qt
import time
from multicamera_systems.tfake import ThermalLandmarks
from neurovc.util.IO_util import map_temp, normalize_color, draw_landmarks
import cv2
from collections import deque
from .base_workers import LandmarkWorker


_mouth_boundary = [203, 98, 97, 2, 326, 327, 423, 426, 436, 432,
                   422, 424, 418, 421, 200, 201, 194, 204, 202, 138, 216, 212, 206]
_eye_left_boundary = [46, 53, 52, 65, 55, 193, 245, 128, 121, 120, 119,
                      118, 117, 111, 35, 124]
_eye_right_boundary = [285, 417, 465, 357, 350, 349, 348, 347, 346,
                       340, 265, 353, 276, 283, 282, 295]


class ThermalLandmarkVisualizer(LandmarkWorker):
    new_frame = pyqtSignal(np.ndarray, float, float)
    _lm_mode = "visible"

    def __init__(self):
        LandmarkWorker.__init__(self)

    def run(self):
        ts_baseline = -1
        while True:
            for (i, cam) in enumerate(self._cams):
                if len(cam) < 1:
                    continue
                try:
                    (n_frame, ts, frame) = cam[-1]
                    (n_frame, ts, landmarks) = self._landmarkers[i][-1]
                except:
                    time.sleep(1/300)
                    continue
                if ts_baseline == -1:
                    ts_baseline = ts

                frame = normalize_color(frame, color_map=cv2.COLORMAP_INFERNO)
                if self._lm_mode == "visible":
                    frame = draw_landmarks(frame, landmarks)

                self.new_frame.emit(frame, 0, ts - ts_baseline)

            time.sleep(1/300)

    @pyqtSlot(object)
    def control_command(self, event):
        command = event.key()
        if command == Qt.Key.Key_Down or command == Qt.Key.Key_Up:
            self._lm_mode = "visible" if self._lm_mode == "invisible" else "invisible"


class TemperatureWorker(LandmarkWorker):
    change_eye_temp = pyqtSignal(float, float)
    change_mouth_temp = pyqtSignal(float, float)
    change_ear_signal = pyqtSignal(float, float)

    def __init__(self):
        LandmarkWorker.__init__(self)
        self.respiration = []
        self.temperature = []

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

            for (i, cam) in enumerate(self._cams):
                if len(cam) < 1:
                    continue
                try:
                    (n_frame, ts_thermal, frame) = cam[-1]
                    (n_frame, ts, pose_mapped) = self._landmarkers[i][-1]
                except:
                    time.sleep(1/300)
                    continue

                if -1 in pose_mapped:
                    continue

                if ts_baseline == -1:
                    ts_baseline = ts_thermal

                thermal_raw = frame.copy()

                temps = []
                try:
                    mask = get_thermal_mask(frame, pose_mapped[_mouth_boundary])
                    mouth = np.sum(thermal_raw[mask != 0]) / (0.000001 + np.sum(mask != 0))
                    mask1 = get_thermal_mask(frame, pose_mapped[_eye_left_boundary])
                    mask2 = get_thermal_mask(frame, pose_mapped[_eye_right_boundary])
                except:
                    print("Failed to extract temperatures...")
                    continue
                mask1[mask2 != 0] = 255
                mask1[mask2 != 0] = 255
                try:
                    eye = np.max(thermal_raw[mask1 != 0])
                except:
                    eye = 0

                self.change_eye_temp.emit(eye, ts_thermal - ts_baseline)
                self.change_mouth_temp.emit(mouth, ts_thermal - ts_baseline)

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
