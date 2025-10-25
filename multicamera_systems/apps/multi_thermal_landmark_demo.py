from multicamera_systems.cameras import ThermalVideoGrabber
from multicamera_systems.ui import VideoWidget
from multicamera_systems.pipeline import MultiThermalLandmarkWorker, ThermalLandmarkVisualizer
from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtCore import pyqtSignal, Qt
import PyQt6.QtWidgets as qtw
import sys
import time


class MultiThermalDemo(QWidget):
    key_press_signal = pyqtSignal(object)

    def __init__(self, thermal):
        super().__init__()
        self.setWindowTitle("Thermal Camera - Multi Landmarks")

        thermal_worker = MultiThermalLandmarkWorker(n_landmarks=70)
        thermal_worker.add_cam(thermal)
        thermal_worker.start()
        time.sleep(1)

        thermal_viz = ThermalLandmarkVisualizer()
        thermal_viz.add_cam(thermal_worker)
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
    vid_path = "C:\\Users\\Philipp\\Documents\\projects\\2025\\camera_documentation\\qt_camera_systems\\test_seq1_green.mp4"
    thermal = ThermalVideoGrabber(vid_path)
    thermal.start()

    app = QApplication(sys.argv)
    demo = MultiThermalDemo(thermal)
    demo.show()

    sys.exit(app.exec())
