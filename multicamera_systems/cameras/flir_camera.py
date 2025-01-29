# ------------------------------------------------------------------------------------
# AUTHOR & CONTEXT DISCLAIMER
#
# Author: Philipp Flotho (philipp.flotho[at]uni-saarland.de), FLIR Systems
#
# For citation, please refer to the project README.
#
# =============================================================================
# Copyright (c) 2001-2023 FLIR Systems, Inc. All Rights Reserved.
#
# This software is the confidential and proprietary information of FLIR
# Integrated Imaging Solutions, Inc. ("Confidential Information"). You
# shall not disclose such Confidential Information and shall use it only in
# accordance with the terms of the license agreement you entered into
# with FLIR Integrated Imaging Solutions, Inc. (FLIR).
#
# FLIR MAKES NO REPRESENTATIONS OR WARRANTIES ABOUT THE SUITABILITY OF THE
# SOFTWARE, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE, OR NON-INFRINGEMENT. FLIR SHALL NOT BE LIABLE FOR ANY DAMAGES
# SUFFERED BY LICENSEE AS A RESULT OF USING, MODIFYING OR DISTRIBUTING
# THIS SOFTWARE OR ITS DERIVATIVES.
# =============================================================================
# ------------------------------------------------------------------------------------

try:
    import torch
    import PySpin
    FLIR_AVAILABLE = True
except ImportError:
    FLIR_AVAILABLE = False
from multicamera_systems.cameras import GenericCamera
from time import time
from PyQt6.QtCore import QThread, pyqtSignal
import numpy as np


def _flir_check():
    if not FLIR_AVAILABLE:
        raise ImportError("PySpin is not available. Please install it to use FLIR cameras.")


class ThermalGrabber(QThread):
    new_frame = pyqtSignal(np.ndarray, float, float)

    def __init__(self, i=0):
        QThread.__init__(self)
        _flir_check()
        self.__cam_id = i
        self.cam = None
        self.n_frame = 0
        self.__cam = None
        self.__callbacks = []
        # self.frame_deque = deque()

    def run(self):
        system = PySpin.System.GetInstance()

        # Get current library version
        version = system.GetLibraryVersion()
        print('Library version: %d.%d.%d.%d' % (version.major, version.minor, version.type, version.build))

        cam_list = system.GetCameras()

        num_cameras = cam_list.GetSize()

        print('Number of cameras detected: %d' % num_cameras)

        self.__cam = cam_list[self.__cam_id]


        print('*** IMAGE ACQUISITION ***\n')
        try:
            result = True

            nodemap_tldevice = self.__cam.GetTLDeviceNodeMap()

            self.__cam.Init()

            nodemap = self.__cam.GetNodeMap()

            # Set acquisition mode to continuous
            #
            #  *** NOTES ***
            #  Because the example acquires and saves 10 images, setting acquisition
            #  mode to continuous lets the example finish. If set to single frame
            #  or multiframe (at a lower number of images), the example would just
            #  hang. This would happen because the example has been written to
            #  acquire 10 images while the camera would have been programmed to
            #  retrieve less than that.
            #
            #  Setting the value of an enumeration node is slightly more complicated
            #  than other node types. Two nodes must be retrieved: first, the
            #  enumeration node is retrieved from the nodemap; and second, the entry
            #  node is retrieved from the enumeration node. The integer value of the
            #  entry node is then set as the new value of the enumeration node.
            #
            #  Notice that both the enumeration and the entry nodes are checked for
            #  availability and readability/writability. Enumeration nodes are
            #  generally readable and writable whereas their entry nodes are only
            #  ever readable.
            #
            #  Retrieve enumeration node from nodemap

            # In order to access the node entries, they have to be casted to a pointer type (CEnumerationPtr here)
            node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
            node_temp_mode = PySpin.CEnumerationPtr(nodemap.GetNode('IRFormat'))
            if not PySpin.IsAvailable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
                print('Unable to set acquisition mode to continuous (enum retrieval). Aborting...')
                return False

            # Retrieve entry node from enumeration node
            node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
            node_temp_mode_linear = node_temp_mode.GetEntryByName('TemperatureLinear10mK')
            if not PySpin.IsAvailable(node_acquisition_mode_continuous) or not PySpin.IsReadable(
                    node_acquisition_mode_continuous):
                print('Unable to set acquisition mode to continuous (entry retrieval). Aborting...')
                return False

            # Retrieve integer value from entry node
            temp_mode = node_temp_mode_linear.GetValue()
            acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()

            # Set integer value from entry node as new value of enumeration node
            node_temp_mode.SetIntValue(temp_mode)
            node_acquisition_mode.SetIntValue(acquisition_mode_continuous)

            print('Acquisition mode set to continuous...')

            #  Begin acquiring images
            #
            #  *** NOTES ***
            #  What happens when the camera begins acquiring images depends on the
            #  acquisition mode. Single frame captures only a single image, multi
            #  frame catures a set number of images, and continuous captures a
            #  continuous stream of images. Because the example calls for the
            #  retrieval of 10 images, continuous mode has been set.
            #
            #  *** LATER ***
            #  Image acquisition must be ended when no more images are needed.

            #  Retrieve device serial number for filename
            #
            #  *** NOTES ***
            #  The device serial number is retrieved in order to keep cameras from
            #  overwriting one another. Grabbing image IDs could also accomplish
            #  this.
            device_serial_number = ''
            node_device_serial_number = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))
            self.__cam.BeginAcquisition()

            #for i in range(100):
            #    image_result = self.__cam.GetNextImage(1000)
            #    if not image_result.IsIncomplete():
            #        print("broken frame " + str(i))
            #    else:
            #        print("working frame " + str(i))

        except PySpin.SpinnakerException as ex:
            print('Error: %s' % ex)
            return False
        print("initialized")
        while True:
            #try:
            image_result = self.__cam.GetNextImage(1000)
            if not image_result.IsIncomplete():
                frame = image_result.GetNDArray()
                self.n_frame += 1
            else:
                continue
            #except:
            #    continue
            image_result.Release()
            ts = time() * 1000000

            # print("got frame " + str(self.n_frame))

            if frame is None:
                continue
            self.__callback(self.n_frame, ts, frame)
            # result_tuple = (self.n_frame, ts, frame)
            self.new_frame.emit(frame, ts, self.n_frame)
            # self.frame_deque.append(result_tuple)

    def add_callback(self, callback):
        self.__callbacks.append(callback)

    def remove_callback(self, callback):
        self.__callbacks.remove(callback)

    def __callback(self, n_frame, ts, frame):
        for f in self.__callbacks:
            f(n_frame, ts, frame)


class ThermalCamera(GenericCamera):
    def __init__(self, i=0):
        self.__cam = None
        self.__cam_id = i

        self.n_frame = 0

    def start_acquisition(self):
        # Thermal camera initialization (Spinnaker example):
        # Retrieve singleton reference to system object
        system = PySpin.System.GetInstance()

        # Get current library version
        version = system.GetLibraryVersion()
        print('Library version: %d.%d.%d.%d' % (version.major, version.minor, version.type, version.build))

        # Retrieve list of cameras from the system
        cam_list = system.GetCameras()

        num_cameras = cam_list.GetSize()

        print('Number of cameras detected: %d' % num_cameras)

        #if self.__cam_id > num_cameras:
        #    return

        self.__cam = cam_list[self.__cam_id]
        self.__cam.Init()

        nodemap_tldevice = self.__cam.GetTLDeviceNodeMap()
        nodemap = self.__cam.GetNodeMap()


        print('*** IMAGE ACQUISITION ***\n')
        try:
            result = True

            # Set acquisition mode to continuous
            #
            #  *** NOTES ***
            #  Because the example acquires and saves 10 images, setting acquisition
            #  mode to continuous lets the example finish. If set to single frame
            #  or multiframe (at a lower number of images), the example would just
            #  hang. This would happen because the example has been written to
            #  acquire 10 images while the camera would have been programmed to
            #  retrieve less than that.
            #
            #  Setting the value of an enumeration node is slightly more complicated
            #  than other node types. Two nodes must be retrieved: first, the
            #  enumeration node is retrieved from the nodemap; and second, the entry
            #  node is retrieved from the enumeration node. The integer value of the
            #  entry node is then set as the new value of the enumeration node.
            #
            #  Notice that both the enumeration and the entry nodes are checked for
            #  availability and readability/writability. Enumeration nodes are
            #  generally readable and writable whereas their entry nodes are only
            #  ever readable.
            #
            #  Retrieve enumeration node from nodemap

            # In order to access the node entries, they have to be casted to a pointer type (CEnumerationPtr here)
            node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
            if not PySpin.IsAvailable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
                print('Unable to set acquisition mode to continuous (enum retrieval). Aborting...')
                return False

            # Retrieve entry node from enumeration node
            node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
            if not PySpin.IsAvailable(node_acquisition_mode_continuous) or not PySpin.IsReadable(
                    node_acquisition_mode_continuous):
                print('Unable to set acquisition mode to continuous (entry retrieval). Aborting...')
                return False

            # Retrieve integer value from entry node
            acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()

            # Set integer value from entry node as new value of enumeration node
            node_acquisition_mode.SetIntValue(acquisition_mode_continuous)

            print('Acquisition mode set to continuous...')

            #  Begin acquiring images
            #
            #  *** NOTES ***
            #  What happens when the camera begins acquiring images depends on the
            #  acquisition mode. Single frame captures only a single image, multi
            #  frame catures a set number of images, and continuous captures a
            #  continuous stream of images. Because the example calls for the
            #  retrieval of 10 images, continuous mode has been set.
            #
            #  *** LATER ***
            #  Image acquisition must be ended when no more images are needed.

            #  Retrieve device serial number for filename
            #
            #  *** NOTES ***
            #  The device serial number is retrieved in order to keep cameras from
            #  overwriting one another. Grabbing image IDs could also accomplish
            #  this.
            device_serial_number = ''
            node_device_serial_number = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))

            self.__cam.BeginAcquisition()

        except PySpin.SpinnakerException as ex:
            print('Error: %s' % ex)
            return False

    def get_image(self):
        try:
            image_result = self.__cam.GetNextImage(1)
            if not image_result.IsIncomplete():
                img = image_result.GetNDArray()
                img = img[:480, :640]
                self.n_frame += 1
                return self.n_frame, time(), img
            else:
                return None, None, None
        except PySpin.SpinnakerException as ex:
            print('Error: %s' % ex)
            return None, None, None

    def __del__(self):
        if self.__cam is not None:
            self.__cam.EndAcquisition()
        super(ThermalCamera, self).__del__()


if __name__ == "__main__":
    import cv2
    from neurovc.util.IO_util import normalize_color

    cam = ThermalGrabber()
    cam.start()
