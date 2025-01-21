# VI-Screen demo requiring one thermal and one ximea camera
from PyQt6 import QtWidgets as qtw
from PyQt6.QtWidgets import QApplication, QWidget
import sys
from multicamera_systems.pipeline import HeartRateWorker
from multicamera_systems.ui import (
    MrPlotterRows, SyncVideoLine, VideoSwitcher, TextBlock
)
from multicamera_systems.cameras import (
    XimeaCamera, ThermalGrabber, FrameGrabber
)


class VIScreener(QWidget):
    def __init__(self, left, right, thermal):
        super().__init__()
        self.setWindowTitle("Thermal Camera - RoI Temperature")

        mapping_worker = TemperatureWorker(left, right, thermal)
        mapping_worker.start()
        thermal_viz = ThermalVizThread()
        thermal_viz.add_cam(thermal)
        thermal_viz.start()

        # momag_worker = MotionMagnificationThread()
        # momag_worker.add_cam(left)
        # momag_worker.start()

        hr_worker = HeartRateWorker()
        hr_worker.add_cam(left)
        hr_worker.start()

        self.resize(1000, 1000)

        self.video_line = SyncVideoLine([
            mapping_worker, left, right])
        self.video_line.resize(1000, 500)
        self.video_switcher = VideoSwitcher(
            [thermal_viz, mapping_worker, left, right, hr_worker]
        )
        # self.video_switcher.key_press_signal.connect(momag_worker.control_command)

        self.plotter_rows = MrPlotterRows([
            (hr_worker.change_hr_real_signal, "Real Heart Rate", ["Heart Rate", "bpm"]),
            (hr_worker.change_hr_signal, "POS", ["POS", ""]),
            (mapping_worker.change_ear_signal, "Eye open / closed signal", ["", "mm"]),
            (mapping_worker.change_eye_temp, "Periorbital Temperature", ["Temperature", "°C"]),
            (mapping_worker.change_mouth_temp, "Mouth Temperature", ["Temperature", "°C"])
        ])
        self.plotter_rows.resize(100, 100)
        self.text = TextBlock([
            (hr_worker.change_hr_real_signal, "Heart Rate", "bpm"),
            (mapping_worker.change_eye_temp, "Temperature", "°C"),
            (mapping_worker.change_blinking_rate, "Blinking Rate", "bpm"),
        ])

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
    cam_left = XimeaCamera(get_left_id())
    cam_right = XimeaCamera(get_right_id())

    left = FrameGrabber(cam_left)
    right = FrameGrabber(cam_right)
    thermal = ThermalGrabber(_get_thermal_id())

    left.start()
    thermal.start()
    right.start()

    app = QApplication(sys.argv)
    a = VIScreener(left, right, thermal)
    a.show()

    sys.exit(app.exec())
    sys.exit(app.exec())


if __name__ == "__main__":
    vi_screen_main()
