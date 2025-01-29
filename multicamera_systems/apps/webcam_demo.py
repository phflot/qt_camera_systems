import sys
import PyQt6.QtWidgets as qtw
from PyQt6.QtWidgets import QWidget, QApplication

from multicamera_systems.cameras import WebcamCamera, FrameGrabber
from multicamera_systems.ui import MrPlotterRows, SyncVideoLine, VideoSwitcher
from multicamera_systems.pipeline import DataIOThread, RGBPOSLandmarkThread


class RGBDemo(QWidget):
    def __init__(self, cams):
        super().__init__()
        self.setWindowTitle("Thermal Camera - RoI Temperature")
        self.resize(1000, 1000)

        self.video_line = SyncVideoLine(cams)
        self.video_line.resize(1000, 500)
        self.video_switcher = VideoSwitcher(cams)
        self.plotter_rows = MrPlotterRows([
            (cams[0].change_plot_signal, "POS", ["POS", ""]),
            (cams[0].change_ear_signal, "Eye open / closed signal", ["", "mm"])
        ])
        self.plotter_rows.resize(100, 100)

        grid = qtw.QGridLayout()
        grid.addWidget(self.video_line, 0, 0, 1, 2)
        grid.addWidget(self.video_switcher, 1, 0)
        grid.addWidget(self.plotter_rows, 1, 1)
        grid.setRowStretch(0, 1)  # Video line takes twice the space
        grid.setRowStretch(1, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        self.setLayout(grid)
        self.setStyleSheet("background-color: black;")


if __name__ == "__main__":

    cam_left = WebcamCamera()

    left = FrameGrabber(cam_left)

    dataIO_worker = DataIOThread()
    visualization_worker = RGBPOSLandmarkThread()
    visualization_worker.add_cam(left)

    left.start()
    visualization_worker.start()

    app = QApplication(sys.argv)

    a = RGBDemo([visualization_worker, visualization_worker, visualization_worker, visualization_worker, visualization_worker, visualization_worker])
    a.show()

    sys.exit(app.exec())