from multicamera_systems.cameras import ThermalGrabber
from multicamera_systems.ui import VideoWidget
from multicamera_systems.pipeline import ThermalVizThread
from PyQt6.QtWidgets import QApplication, QWidget
import PyQt6.QtWidgets as qtw
import sys


class VitalSignMomagTest2(QWidget):
    def __init__(self, thermal):
        super().__init__()
        self.setWindowTitle("Thermal Camera - RoI Temperature")

        thermal_viz = ThermalVizThread()
        thermal_viz.add_cam(thermal)
        thermal_viz.start()

        self.resize(1000, 1000)

        self.video = VideoWidget(thermal_viz)
        self.video.resize(1000, 500)

        grid = qtw.QGridLayout()
        grid.addWidget(self.video, 0, 0, 1, 1)
        self.setLayout(grid)
        self.setStyleSheet("background-color: black;")


if __name__ == "__main__":
    thermal = ThermalGrabber(0)
    thermal.start()

    app = QApplication(sys.argv)
    a = VitalSignMomagTest2(thermal)
    a.show()

    sys.exit(app.exec())
