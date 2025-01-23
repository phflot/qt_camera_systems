from .base_workers import MultimodalWorker
from PyQt6.QtCore import pyqtSignal
import numpy as np
import mediapipe as mp
from multicamera_systems.tfake import ThermalLandmarks
from neurovc.util import map_temp
import time


class ThermalLandmarkWorker(MultimodalWorker):
    new_frame = pyqtSignal(np.ndarray, float, float)
    new_landmarks = pyqtSignal(np.ndarray, float, float)

    def __init__(self):
        MultimodalWorker.__init__(self)
        self.landmarker = ThermalLandmarks()

    def run(self):
        ts_baseline = -1
        counter = 0
        while True:
            for (i, cam) in enumerate(self._cams):
                if len(cam) < 1:
                    continue
                (n_frame, ts, frame) = cam[-1]
                if ts_baseline == -1:
                    ts_baseline = ts

                frame_raw = map_temp(frame, "A655")
                landmarks = self.landmarker.process(frame_raw)[0]
                self.new_frame.emit(frame, counter, ts - ts_baseline)
                self.new_landmarks.emit(landmarks, counter, ts - ts_baseline)
                counter += 1

            time.sleep(1/30)


class MediapipeLandmarkWorker(MultimodalWorker):
    new_frame = pyqtSignal(np.ndarray, float, float)
    new_landmarks = pyqtSignal(np.ndarray, float, float)

    def __init__(self):
        MultimodalWorker.__init__(self)

        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=0.25
        )

    def run(self):
        starting_time = -1
        counter = 0
        while True:
            for (i, cam) in enumerate(self._cams):
                if len(cam) < 1:
                    continue
                (n_frame, ts, frame) = cam[-1]
                if starting_time == -1:
                    starting_time = ts

                n, m = frame.shape[0:2]
                frame8b = frame.copy()
                landmarks = self.face_mesh.process(frame8b).multi_face_landmarks

                if landmarks is not None:
                    l = np.array(
                        [[m * lm.x, n * lm.y, lm.z] for lm in landmarks[0].landmark])
                    self.new_frame.emit(frame, counter, ts - starting_time)
                    self.new_landmarks.emit(l, counter, ts - starting_time)
                    counter += 1
