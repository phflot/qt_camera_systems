from multicamera_systems.cameras import ThermalGrabber
from multicamera_systems.ui import VideoWidget
from multicamera_systems.pipeline import ThermalLandmarkWorker, ThermalLandmarkVisualizer
from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtCore import pyqtSignal, Qt
import PyQt6.QtWidgets as qtw
import sys
import time


class VIScreenerMomagTest2(QWidget):
    key_press_signal = pyqtSignal(object)

    def __init__(self, thermal):
        super().__init__()
        self.setWindowTitle("Thermal Camera - RoI Temperature")

        thermal_lm = ThermalLandmarkWorker()
        thermal_lm.add_cam(thermal)
        thermal_lm.start()
        time.sleep(1)

        thermal_viz = ThermalLandmarkVisualizer()
        thermal_viz.add_cam(thermal_lm)
        thermal_viz.start()
        self.key_press_signal.connect(thermal_viz.control_command)

        self.resize(1000, 1000)

        self.video = VideoWidget(thermal_viz)
        self.video.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.video.resize(1000, 500)

        grid = qtw.QGridLayout()
        grid.addWidget(self.video, 0, 0, 1, 1)
        self.setLayout(grid)
        self.setStyleSheet("background-color: black;")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def keyPressEvent(self, event):
        self.key_press_signal.emit(event)
        super().keyPressEvent(event)


if __name__ == "__main__":
    thermal = ThermalGrabber(0)
    thermal.start()

    app = QApplication(sys.argv)
    a = VIScreenerMomagTest2(thermal)
    a.show()

    sys.exit(app.exec())
