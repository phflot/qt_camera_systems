# ------------------------------------------------------------------------------------
# AUTHOR STATEMENT
#
# Author: Philipp Flotho (philipp.flotho[at]uni-saarland.de)
#
# For citation, please refer to the project README.
# If you believe any confidential or proprietary content is included, please notify me.
#
# Copyright (c) 2025, Philipp Flotho
# ------------------------------------------------------------------------------------

import time
from collections import deque
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal


class FrameGrabber(QThread):
    new_frame = pyqtSignal(np.ndarray, float, float)

    def __init__(self, cam):
        QThread.__init__(self)
        self.cam = cam
        self.__callbacks = []
        self.frame_deque = deque()

    def run(self):
        self.cam.start_acquisition()
        while True:
            n_frame, ts, frame = self.cam.get_image()
            if frame is None:
                continue
            self.new_frame.emit(frame, n_frame, ts)

    def add_callback(self, callback):
        self.__callbacks.append(callback)

    def remove_callback(self, callback):
        self.__callbacks.remove(callback)

    def __callback(self, n_frame, ts, frame):
        for f in self.__callbacks:
            f(n_frame, ts, frame)