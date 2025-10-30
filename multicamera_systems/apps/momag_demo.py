from PyQt6 import QtWidgets as qtw
from PyQt6.QtWidgets import QWidget, QApplication
from PyQt6.QtCore import pyqtSignal, Qt
import sys
from multicamera_systems.cameras import XimeaCamera, WebcamCamera, FrameGrabber
from multicamera_systems.pipeline import MotionMagnificationThread
from multicamera_systems.ui import SyncVideoLine, VideoSwitcher


class VitalSignMomagTest2(QWidget):
    key_press_signal = pyqtSignal(object)

    def __init__(self, left):
        super().__init__()
        self.setWindowTitle("Thermal Camera - RoI Temperature")

        momag_worker = MotionMagnificationThread()
        momag_worker.add_cam(left)
        momag_worker.start()

        self.resize(1000, 1000)

        self.video_line = SyncVideoLine([
            left, momag_worker])
        self.video_line.resize(1000, 500)
        self.video_switcher = VideoSwitcher([
        left, momag_worker], Qt.FocusPolicy.NoFocus)

        self.key_press_signal.connect(momag_worker.control_command)
        self.key_press_signal.connect(self.video_switcher.keyPressEvent)

        grid = qtw.QGridLayout()
        grid.addWidget(self.video_line, 0, 0, 1, 1)
        grid.addWidget(self.video_switcher, 1, 0)
        # grid.addWidget(self.video_switcher, 1, 0)
        # grid.setRowStretch(0, 1)
        # grid.setRowStretch(1, 1)
        # grid.setRowStretch(2, 1)
        # grid.setColumnStretch(0, 1)
        # grid.setColumnStretch(1, 1)
        self.setLayout(grid)
        self.setStyleSheet("background-color: black;")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def keyPressEvent(self, event):
        self.key_press_signal.emit(event)
        super().keyPressEvent(event)


def momag_webcam():
    cam = WebcamCamera()

    left = FrameGrabber(cam)
    left.start()

    app = QApplication(sys.argv)
    a = VitalSignMomagTest2(left)
    a.show()

    sys.exit(app.exec())


def momag_ximea():

    # set the ximea camera user id in the ximea tool
    # set to None to use the first camera
    # set to int to use the camera with the corresponding index
    cam = XimeaCamera()

    left = FrameGrabber(cam)
    left.start()

    app = QApplication(sys.argv)
    a = VitalSignMomagTest2(left)
    a.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    momag_webcam()
    # momag_ximea()
