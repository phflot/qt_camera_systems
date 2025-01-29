# ------------------------------------------------------------------------------------
# AUTHOR STATEMENT
#
# Author: Philipp Flotho (philipp.flotho[at]uni-saarland.de)
#
# For citation, please refer to the project README.
# If you believe any confidential or proprietary content is included, please notify me.
#
# Copyright (c) 2025, Philipp Flotho
# ------------------------------------------------------------------------------------

import numpy as np
import PyQt6.QtWidgets as qtw
from PyQt6.QtWidgets import QWidget, QLabel
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import pyqtSlot, Qt, pyqtSignal


class VideoWidget(QWidget):
    def __init__(self, camera):
        super(VideoWidget, self).__init__()
        self.camera = camera
        self.camera.new_frame.connect(self.update_image)
        self.image_label = QLabel(self)
        # self.image_label.setScaledContents(True)

    def resizeEvent(self, event):
        """Handle the resizing of the widget, update pixmap to fit the QLabel."""
        width = event.size().width()
        height = event.size().height()
        if self.image_label.pixmap() is not None:
            self.image_label.resize(width, height)
            scaled_pixmap = self.image_label.pixmap().scaled(width, height)
            self.image_label.setPixmap(scaled_pixmap)
        super(VideoWidget, self).resizeEvent(event)

    @pyqtSlot(np.ndarray, float, float)
    def update_image(self, cv_img, n_frame, timestamp):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)
        #self.update_plot(temperature, timestamp)

    def convert_cv_qt(self, img):
        """Convert from an opencv image to QPixmap"""
        height, width, n_channels = img.shape
        bytes_per_line = n_channels * width
        convert_to_Qt_format = QImage(img.data, width, height, bytes_per_line, QImage.Format.Format_BGR888)
        p = convert_to_Qt_format.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio)
        return QPixmap.fromImage(p)


class SyncVideoLine(QWidget):
    '''Line on top with a frame from each camera'''
    key_press_signal = pyqtSignal(object)

    def __init__(self, cameras: list, focus_policy=Qt.FocusPolicy.NoFocus):
        super(SyncVideoLine, self).__init__()
        self.setFocusPolicy(focus_policy)

        self.layout = qtw.QGridLayout(self)

        self.width = 1000
        self.height = 500
        self.resize(self.width, self.height)

        self.num_cameras = len(cameras)
        widget_width = self.width // self.num_cameras

        for i, camera in enumerate(cameras):
            video_widget = VideoWidget(camera)
            video_widget.resize(widget_width, self.height)
            self.layout.addWidget(video_widget, 0, i)

    def resizeEvent(self, event):
        self.width = event.size().width()
        self.height = event.size().height()
        num_cameras = self.num_cameras
        widget_width = self.width // num_cameras
        for i in range(num_cameras):
            widget = self.layout.itemAt(i).widget()
            widget.resize(widget_width, self.height)

    def keyPressEvent(self, event):
        self.key_press_signal.emit(event)


class VideoSwitcher(QWidget):
    key_press_signal = pyqtSignal(object)

    def __init__(self, cameras: list, focus_policy=Qt.FocusPolicy.StrongFocus):
        super(VideoSwitcher, self).__init__()
        self.setFocusPolicy(focus_policy)

        self.cameras = cameras
        self.current_camera = 0

        self.video_widget = VideoWidget(self.cameras[self.current_camera])
        self.video_widget.resize(1000, 1000)

        self.setLayout(qtw.QVBoxLayout())
        self.layout().addWidget(self.video_widget)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Right:
            self.cameras[self.current_camera].new_frame.disconnect(self.video_widget.update_image)
            self.current_camera = (self.current_camera + 1) % len(self.cameras)
            self.cameras[self.current_camera].new_frame.connect(self.video_widget.update_image)
        elif event.key() == Qt.Key.Key_Left:
            self.cameras[self.current_camera].new_frame.disconnect(self.video_widget.update_image)
            self.current_camera = (self.current_camera - 1) % len(self.cameras)
            self.cameras[self.current_camera].new_frame.connect(self.video_widget.update_image)
        else:
            self.key_press_signal.emit(event)
