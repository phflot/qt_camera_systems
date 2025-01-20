import PIL
from PIL import ImageDraw
from scipy.spatial import ConvexHull
from multicamera_systems.pipeline import MultimodalWorker
import numpy as np
from PyQt6.QtCore import pyqtSignal
import time
from neurovc.util.IO_util import map_temp, normalize_color
import cv2
import mediapipe as mp


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
