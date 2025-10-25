from PyQt6.QtCore import QThread, pyqtSignal
import numpy as np
import cv2
import time


class ThermalVideoGrabber(QThread):
    new_frame = pyqtSignal(np.ndarray, float, float)

    def __init__(self, path, fps=30.0):
        QThread.__init__(self)
        self.path = path
        self.fps = fps
        self._cap = None
        self._frame_idx = 0

    def _ensure_capture(self):
        if self._cap is None:
            self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            self._cap.open(self.path)

    def run(self):
        delay = 0 if self.fps <= 0 else 1.0 / self.fps
        while True:
            self._ensure_capture()
            ret, frame = self._cap.read()
            if not ret:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self._frame_idx = 0
                continue
            if frame.ndim == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = frame.astype(np.float32)
            frame = 20.0 + (frame / 255.0) * 18.0
            ts = float(time.time() * 1000000.0)
            self.new_frame.emit(frame, float(self._frame_idx), ts)
            self._frame_idx += 1
            if delay > 0:
                time.sleep(delay)
