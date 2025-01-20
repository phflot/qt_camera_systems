from PyQt6 import QtWidgets as qtw
from PyQt6.QtWidgets import QWidget, QApplication
import sys


class VIScreenerMomagTest(QWidget):
    def __init__(self, camera):
        super().__init__()
        self.setWindowTitle("Thermal Camera - RoI Temperature")

        momag_worker = MotionMagnificationThread()
        momag_worker.add_cam(camera)
        momag_worker.start()

        self.resize(1000, 1000)

        self.video_line = SyncVideoLine([
            camera, momag_worker])
        self.video_line.resize(1000, 500)
        self.video_switcher = VideoSwitcher([
            momag_worker, camera])

        self.video_switcher.key_press_signal.connect(momag_worker.control_command)

        grid = qtw.QGridLayout()
        grid.addWidget(self.video_line, 0, 0, 1, 3)
        grid.addWidget(self.video_switcher, 1, 0)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)
        grid.setRowStretch(2, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        self.setLayout(grid)
        self.setStyleSheet("background-color: black;")


def momag_webcam():
    cam = WebcamCamera()
    left = FrameGrabber(cam)
    left.start()

    app = QApplication(sys.argv)
    a = VIScreenerMomagTest(left)
    a.show()

    sys.exit(app.exec())


def momag_ximea():
    pass


if __name__ == "__main__":
    momag_webcam()
