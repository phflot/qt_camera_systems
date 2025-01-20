import numpy as np
import pyqtgraph as pg
from PyQt6 import QtWidgets as qtw, QtGui
from PyQt6.QtWidgets import QWidget, QLabel
from PyQt6.QtCore import pyqtSlot, Qt
from scipy.signal import savgol_filter
from collections import deque


class TextWidget(QWidget):
    def __init__(self, signal, name, unit):
        super(TextWidget, self).__init__()
        self.name = name
        self.unit = unit
        self.text_label = QLabel(f"{self.name}: -- {self.unit}")

        self.text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.text_label.setFont(QtGui.QFont("Arial", 20))
        self.text_label.setStyleSheet("color: white;")
        self.layout = qtw.QVBoxLayout()
        self.layout.addWidget(self.text_label)
        self.setLayout(self.layout)
        self.data_deque = deque(maxlen=50)
        signal.connect(self.update_value)

    @pyqtSlot(float, float)
    def update_value(self, value, timestamp):
        self.data_deque.append(value)
        self.text_label.setText(f"{self.name}: {np.mean(value):.2f} {self.unit}")


class TextBlock(QWidget):
    def __init__(self, signals: list):
        super(TextBlock, self).__init__()

        self.layout = qtw.QGridLayout(self)

        self.width = 500
        self.height = 1000
        self.resize(self.width, self.height)

        self.num_cameras = len(signals)
        widget_height = self.height // self.num_cameras

        for i, camera in enumerate(signals):
            text_widget = TextWidget(*camera)
            text_widget.resize(self.width, widget_height)
            self.layout.addWidget(text_widget, i, 0)

    def resizeEvent(self, event):
        self.width = event.size().width()
        self.height = event.size().height()
        num_cameras = self.num_cameras
        widget_height = self.height // num_cameras
        for i in range(num_cameras):
            widget = self.layout.itemAt(i).widget()
            widget.resize(self.width, widget_height)


class MrPlotter(QWidget):
    def __init__(self, signal, title="Title", ylabel=["Y1", "Y2"]):
        super(MrPlotter, self).__init__()
        raw_pen = pg.mkPen({'color': (128, 128, 255, 180), 'width': 2.0})
        smooth_pen = pg.mkPen({'color': (255, 255, 0, 255), 'width': 2.0})
        plt_res = 240

        self.graph_widget = []

        self.graph_widget = pg.PlotWidget()
        self.graph_widget.setTitle(title)
        self.graph_widget.setLabel('right', ylabel[0], ylabel[1])
        self.graph_widget.setLabel('bottom', 'Time', 's')
        # self.graph_widget.setYRange(30, 150)
        self.graph_widget.enableAutoRange(axis='y', enable=True)
        self.plot_data = [[0.0 for _ in range(plt_res)] for _ in range(2)]
        # self.processed_plot_data = [[0.0 for _ in range(60)] for _ in range(2)]
        self.curve = self.graph_widget.plot(self.plot_data[0][:plt_res], self.plot_data[1][:plt_res], pen=raw_pen)
        self.curve_2 = self.graph_widget.plot(self.plot_data[0][:plt_res], self.plot_data[1][:plt_res], pen=smooth_pen)
        self.frame_num = 0
        fill_box_layout = qtw.QVBoxLayout()
        fill_box_layout.addWidget(self.graph_widget)
        self.setLayout(fill_box_layout)
        signal.connect(self.update_plot)

    @pyqtSlot(float, float)
    def update_plot(self, temperature, timestamp):
        self.frame_num += 1
        self.plot_data[1].pop(0)
        self.plot_data[1].append(temperature)
        self.plot_data[0].pop(0)
        self.plot_data[0].append(timestamp / 1000000.0)

        smoothed = savgol_filter(self.plot_data[1], 30, 7).tolist()

        self.curve.setData(self.plot_data[0], self.plot_data[1])
        self.curve_2.setData(self.plot_data[0], smoothed)


class MrPlotterRows(QWidget):
    def __init__(self, signals: list):
        super(MrPlotterRows, self).__init__()

        self.layout = qtw.QGridLayout(self)

        self.width = 500
        self.height = 1000
        self.resize(self.width, self.height)

        self.num_cameras = len(signals)
        widget_height = self.height // self.num_cameras

        for i, camera in enumerate(signals):
            plotter = MrPlotter(*camera)
            plotter.resize(self.width, widget_height)
            self.layout.addWidget(plotter, i, 0)

    def resizeEvent(self, event):
        self.width = event.size().width()
        self.height = event.size().height()
        num_cameras = self.num_cameras
        widget_height = self.height // num_cameras
        for i in range(num_cameras):
            widget = self.layout.itemAt(i).widget()
            widget.resize(self.width, widget_height)
