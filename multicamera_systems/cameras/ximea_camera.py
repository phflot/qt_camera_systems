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

from .base_cameras import GenericCamera
import cv2
import time
import numpy as np


try:
    import ximea
    from ximea import xiapi
    from ximea.xiapi import Camera, Image
    XIMEA_AVAILABLE = True
except ImportError:
    XIMEA_AVAILABLE = False


def _ximea_check():
    if not XIMEA_AVAILABLE:
        raise ImportError("The ximea module is not available. Please install it from the ximea API installer.")


def _is_camera_opened(camera):
    try:
        camera.get_param('device_sn')
        return True
    except xiapi.Xi_error:
        return False


def get_right_id():
    return 'right'


def get_nir_id():
    return 'nir'


def init_cam(cam, exp=20000):
    _ximea_check()
    _ximea_check()

    try:
        cam.enable_auto_wb()
    except ximea.xiapi.Xi_error as e:
        print(e)
    cam.set_exposure(exp)
    cam.set_imgdataformat('XI_RGB24')


class XimeaCamera(GenericCamera):
    def __init__(self, user_id=None):
        _ximea_check()
        self.cam = Camera()
        self.user_id = user_id
        if user_id is None:
            for i in range(10):
                if not _is_camera_opened(Camera(i)):
                    user_id = i
                    break
            if user_id is None:
                raise(ValueError("No camera available"))
        if isinstance(user_id, str):
            self.cam.open_device_by("XI_OPEN_BY_USER_ID", user_id)
        else:
            self.cam = Camera(user_id)
            self.cam.open_device()
        init_cam(self.cam)
        if user_id == get_nir_id():
            self.cam.set_imgdataformat('XI_MONO8')
        self.cam.set_param('downsampling_type', 'XI_SKIPPING')
        self.cam.set_param('downsampling', "XI_DWN_2x2")

        self.cam.start_acquisition()
        self.__image = Image()
        self.frame_counter = 0

    def get_image(self):
        try:
            self.cam.get_image(self.__image)
            image = self.__image.get_image_data_numpy()
            ts = 1000000 * time.time() # '* self.__image.tsSec + self.__image.tsUSec
            n_frame = self.frame_counter # self.__image.acq_nframe
            self.frame_counter += 1
        except:
            return (None, None, None)

        if len(image.shape) == 2:
            image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        else:
            image = cv2.resize(image, None, fx=0.5, fy=0.5).astype(float)

        return float(n_frame), float(ts), image.astype(np.uint8)
