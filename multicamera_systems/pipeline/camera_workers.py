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
from PyQt6.QtCore import QThread
from collections import deque


class MultimodalWorker(QThread):
    def __init__(self):
        QThread.__init__(self)
        self._cams = []

    def add_cam(self, cam):
        frame_deque = deque(maxlen=10)
        self._cams.append(frame_deque)
        cam.new_frame.connect(
            lambda frame, n_frame, ts: frame_deque.append((n_frame, ts, frame)))


class TriggerThread(MultimodalWorker):
    def __init__(self, fps=30):
        MultimodalWorker.__init__(self)
        self.__fps = fps

    def run(self):
        while True:
            start = time.time()
            try:
                for c in self._cams:
                    c.cam.set_trigger_software(1)
            except:
                pass
            elapsed = time.time() - start
            sleep_time = (1 / self.__fps) - elapsed
            time.sleep(max(sleep_time, 0))


class DataIOThread(MultimodalWorker):
    def __init__(self):
        MultimodalWorker.__init__(self)

    def run(self):
        old_timestamp = [-1] * len(self._cams)
        old_id = [-1] * len(self._cams)

        while True:
            for (i, cam) in enumerate(self._cams):
                if len(cam) > 1:
                    (n_frame, ts, frame) = cam.popleft()
                    old_id[i] = n_frame
                    old_timestamp[i] = ts
                    del(frame)
            time.sleep(0.001)
