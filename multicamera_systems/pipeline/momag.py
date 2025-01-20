from neurovc.momag import MagnificationTask, AlphaLooper
from PyQt6.QtCore import pyqtSignal, pyqtSlot, Qt
import time
import numpy as np
from multicamera_systems.pipeline import MultimodalWorker

class MotionMagnificationThread(MultimodalWorker):
    new_frame = pyqtSignal(np.ndarray, float, float)

    def __init__(self):
        MultimodalWorker.__init__(self)
        self.last_command = "dynamic"
        self.last_magnifier = 0
        self.warper = MagnificationTask()
        self.looper = AlphaLooper([1, 8], 0.5)

    @pyqtSlot(int)
    def control_command(self, command):
        if command == Qt.Key.Key_Space:
            self.last_command = "static" if self.last_command == "dynamic" else "dynamic"
        if command in [Qt.Key.Key_1, Qt.Key.Key_2, Qt.Key.Key_3, Qt.Key.Key_4, Qt.Key.Key_5]:
            self.last_magnifier = command

    def run(self):
        ts_baseline = -1

        while True:
            cam = self._cams[0]

            if len(cam) < 1:
                continue
            (n_frame, ts, frame) = cam[-1]
            if ts_baseline == -1:
                ts_baseline = ts

            if self.last_command == "static":
                do_continue = True
                while do_continue:
                    magnified = self.warper.get_mag(self.looper())
                    if magnified is None:
                        print("No magnified frame available!")
                        break
                    self.new_frame.emit(magnified, 0, ts - ts_baseline)
                    self.warper.set_magnifier(self.last_magnifier, True)
                    time.sleep(1 / 200)
                    if self.last_command == "dynamic":
                        self.looper.reset()
                        do_continue = False

            self.warper.set_magnifier(self.last_magnifier, True)
            self.warper(frame)
            magnified = self.warper.get_mag(5)

            self.new_frame.emit(magnified, 0, ts - ts_baseline)

            time.sleep(1/200)
