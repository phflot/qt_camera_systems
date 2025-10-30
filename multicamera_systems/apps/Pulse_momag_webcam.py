# minimal demo using a single webcam

# VI-Screen demo requiring one thermal and one ximea camera

from PyQt6 import QtWidgets as qtw
from PyQt6.QtWidgets import QApplication, QWidget
import sys
from multicamera_systems.pipeline import (
    HeartRateWorker, TemperatureWorker, BlinkingRateWorker, MotionMagnificationThread,
    ThermalLandmarkWorker, MediapipeLandmarkWorker, ThermalLandmarkVisualizer
)
from multicamera_systems.ui import (
    MrPlotterRows, SyncVideoLine, VideoSwitcher, TextBlock
)
from multicamera_systems.cameras import (
    WebcamCamera, FrameGrabber
)
import time


class VitalSignApp(QWidget):
    def __init__(self, left):
        super().__init__()
        self.setWindowTitle("Thermal Camera - RoI Temperature")

        rgb_lm = MediapipeLandmarkWorker()
        rgb_lm.add_cam(left)

        momag_worker = MotionMagnificationThread()
        momag_worker.add_cam(left)

        hr_worker = HeartRateWorker()
        hr_worker.add_cam(left)

        ear_worker = BlinkingRateWorker()
        ear_worker.add_cam(rgb_lm)

        print("Starting cameras...")
        left.start()
        rgb_lm.start()
        time.sleep(2)

        print("Starting workers...")
        # ear_worker.start()
        time.sleep(0.5)
        momag_worker.start()
        time.sleep(0.5)
        hr_worker.start()
        time.sleep(5)

        self.resize(1000, 1000)

        print("Initializing video output and plots...")
        self.video_line = SyncVideoLine([
            left, momag_worker])
        self.video_line.resize(1000, 500)
        self.video_switcher = VideoSwitcher(
            [left, momag_worker, hr_worker]
        )
        self.video_switcher.key_press_signal.connect(momag_worker.control_command)

        self.plotter_rows = MrPlotterRows([
            (hr_worker.change_hr_real_signal, "Real Heart Rate", ["Heart Rate", "bpm"]),
            (hr_worker.change_hr_signal, "POS", ["POS", ""]),
            (ear_worker.change_ear_signal, "Eye open / closed signal", ["", ""]),
        ])
        self.plotter_rows.resize(100, 100)
        self.text = TextBlock([
            (hr_worker.change_hr_real_signal, "Heart Rate", "bpm"),
            (ear_worker.change_blinking_rate, "Blinking Rate", "bpm"),
        ])

        print("Initializing UI...")
        grid = qtw.QGridLayout()
        grid.addWidget(self.video_line, 0, 0, 1, 3)
        grid.addWidget(self.text, 0, 3, 1, 1)
        grid.addWidget(self.video_switcher, 1, 0, 2, 2)
        grid.addWidget(self.plotter_rows, 1, 2, 2, 2)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)
        grid.setRowStretch(2, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)
        self.setLayout(grid)
        self.setStyleSheet("background-color: black;")


def vi_screen_main():
    cam_left = WebcamCamera()
    left = FrameGrabber(cam_left)

    app = QApplication(sys.argv)
    a = VitalSignApp(left)
    a.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    vi_screen_main()
