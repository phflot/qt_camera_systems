from collections import deque

from PyQt6.QtCore import QThread


class MultimodalWorker(QThread):
    def __init__(self):
        QThread.__init__(self)
        self._cams = []

    def add_cam(self, cam):
        frame_deque = deque(maxlen=10)
        self._cams.append(frame_deque)
        cam.new_frame.connect(
            lambda frame, n_frame, ts: frame_deque.append((n_frame, ts, frame)))


class LandmarkWorker(QThread):
    def __init__(self):
        QThread.__init__(self)
        self._cams = []
        self._landmarkers = []

    def add_cam(self, cam):
        frame_deque = deque(maxlen=10)
        self._cams.append(frame_deque)
        cam.new_frame.connect(
            lambda frame, n_frame, ts:
            frame_deque.append((n_frame, ts, frame)))

        lm_deque = deque(maxlen=10)
        self._landmarkers.append(lm_deque)
        cam.new_landmarks.connect(
            lambda landmarks, n_frame, ts:
            lm_deque.append((n_frame, ts, landmarks)))