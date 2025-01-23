import time
from PyQt6.QtCore import pyqtSignal

from scipy.signal import find_peaks
import numpy as np
from collections import deque
import mediapipe as mp
import cv2

from multicamera_systems.pipeline.base_workers import MultimodalWorker, LandmarkWorker
from multicamera_systems.util import segment_skin
from neurovc.util.IO_util import draw_landmarks
import math


class BlinkingRateWorker(LandmarkWorker):
    change_blinking_rate = pyqtSignal(float, float)
    change_ear_signal = pyqtSignal(float, float)

    def __init__(self):
        LandmarkWorker.__init__(self)
        self.blinkings = []

    def run(self):
        ears = deque(maxlen=3)
        ears_blinking_rate = deque(maxlen=350)
        ears_timestamps = deque(maxlen=350)
        ts_baseline = 0

        while(True):
            for (i, cam) in enumerate(self._cams):
                if len(cam) < 1:
                    continue
                (n_frame, ts_thermal, frame) = cam[-1]
                (n_frame, ts, points3D) = self._landmarkers[i][-1]

                p1 = 159
                p2 = 145
                p11 = 130
                p21 = 243

                p3 = 386
                p4 = 374
                p31 = 463
                p41 = 359

                def norm(e1, e2):
                    return math.sqrt((e1[0] - e2[0]) ** 2 +
                                     (e1[1] - e2[1]) ** 2)

                def get_ear(e1, e2, e3, e4):
                    return norm(e1, e2) / norm(e3, e4)

                ear = get_ear(points3D[:, p1], points3D[:, p2], points3D[:, p11], points3D[:, p21])
                ear += get_ear(points3D[:, p3], points3D[:, p4], points3D[:, p31], points3D[:, p41])
                ear *= 0.5
                ears.append(ear)
                ears_blinking_rate.append(ear)
                ears_timestamps.append(ts_thermal - ts_baseline)

                ears_tmp = np.array(ears_blinking_rate)
                if len(ears_tmp) > 100:
                    peaks, _ = find_peaks(-ears_tmp)
                    blinking_rate = len(peaks) / (ears_timestamps[-1] - ears_timestamps[0])
                    blinking_rate *= 60
                    self.change_blinking_rate.emit(blinking_rate, ears_timestamps[-1] - ears_timestamps[0])

                ear = np.mean(ears)
                self.change_ear_signal.emit(ear, ts_thermal - ts_baseline)


class HeartRateWorker(MultimodalWorker):
    change_hr_signal = pyqtSignal(float, float)
    change_hr_real_signal = pyqtSignal(float, float)
    new_frame = pyqtSignal(np.ndarray, float, float)

    def __init__(self):
        MultimodalWorker.__init__(self)
        self.hr = None
        self.hr_real = None

    def run(self):
        old_timestamp = [-1] * len(self._cams)
        old_id = [-1] * len(self._cams)

        mean_rgb_frames = []
        time_stamps = []
        hr_values = []
        hr_real_values = []
        hr_real_ts = []

        fps = 15
        starting_time = -1
        counter = 0

        while True:
            for (i, cam) in enumerate(self._cams):
                if len(cam) < 1:
                    continue
                (n_frame, ts, frame) = cam[-1]
                counter += 1
                if starting_time == -1:
                    starting_time = ts

                n, m = frame.shape[0:2]

                if n_frame is None or n_frame == old_id[i]:
                    continue

                frame8b = frame.copy()

                #mask = self.segment_func(frame8b)
                #frame8b = segment_image(frame8b, mask)
                frame8b, mask = segment_skin(frame8b)
                self.new_frame.emit(frame8b, n_frame, ts - starting_time)

                mean_rgb = np.sum(frame8b.astype(float), axis=(0, 1)) / (0.000001 + np.sum(mask.astype(float), axis=(0, 1)))
                mean_rgb_frames.append(mean_rgb)
                time_stamps.append(ts - starting_time)

                if len(mean_rgb_frames) > 100:
                    ts_arr = np.array(time_stamps)
                    # print(ts_arr)
                    time_resampled = np.arange(ts_arr[0], ts_arr[-1], 1000000 / fps, float)
                    idx = np.round(np.interp(time_resampled, ts_arr,
                                   np.arange(0, ts_arr.shape[0]))).astype(int)
                    idx = idx[-100:]

                    start = time.time()
                    signal = _pos_calculation(np.array(mean_rgb_frames)[idx])

                    fourier = np.fft.fft(signal)
                    freq = 60 * np.fft.fftfreq(signal.size, d=1/fps)
                    fourier = fourier[freq >= 40]
                    freq = freq[freq >= 40]
                    fourier = fourier[freq <= 140]
                    freq = freq[freq <= 140]
                    heartrate = freq[np.argmax(fourier)]

                    self.change_hr_signal.emit(heartrate, ts_arr[-1])
                    hr_values.append(heartrate)

                    hr_real_values.append(np.mean(np.array(hr_values)))
                    self.change_hr_real_signal.emit(np.mean(np.array(hr_values)), ts_arr[-1])
                    hr_real_ts.append(ts_arr[-1])

                    self.hr = (ts_arr[idx], signal)
                    self.hr_real = (np.array(hr_real_ts), np.array(hr_real_values))

                    if len(hr_real_ts) > 100:
                        hr_real_ts = list(np.array(hr_real_ts)[-100:])
                        hr_real_values = list(np.array(hr_real_values)[-100:])

                    if len(hr_values) > 300:
                        hr_values = list(np.array(hr_values)[-300:])

                    # freqs, spectrum = _createSpectrum(signal, 5)
                    #print(signal.shape)
                    #print("pos estimation took " + str(time.time() - start))
                    #print("heart rate is " + str(np.mean(np.array(hr_values))))

                    time_stamps = list(ts_arr[-100:])
                    mean_rgb_frames = list(np.array(mean_rgb_frames)[-100:])


                cv2.imshow("segmented", frame8b)
                cv2.waitKey(1)
                #text = "#frame " + str(n_frame) + "\n fps = " + str(fps)
                #cv2.putText(frame8b, text,
                #            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

                # cv2.imshow(str(i), frame8b)
                # cv2.waitKey(1)

                old_id[i] = n_frame
                old_timestamp[i] = ts

            time.sleep(1/300)


class DataFusionThread(MultimodalWorker):
    new_frame = pyqtSignal(np.ndarray, float, float)
    change_plot_signal = pyqtSignal(float, float)
    change_ear_signal = pyqtSignal(float, float)
    def __init__(self):
        MultimodalWorker.__init__(self)
        # self.segment_func = Segmenter()
        self.segment_func = segment_skin
        self.hr = None
        self.hr_real = None
        self.ear = None
        # self.inpainting_func = Inpainter()

    def run(self):
        # example thread that does visualization and computationally heavy fusion / stereo, ...
        old_timestamp = [-1] * len(self._cams)
        old_id = [-1] * len(self._cams)
        initialized = False

        mean_rgb_frames = []
        time_stamps = []
        hr_values = []
        hr_real_values = []
        hr_real_ts = []
        ear_array = []
        ts_array = []

        fps = 15
        starting_time = -1
        counter = 0

        mp_drawing = mp.solutions.drawing_utils
        mp_face_mesh = mp.solutions.face_mesh
        drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=0, color=(0, 0, 0))

        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=0.25)

        selfie = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0)

        while True:
            for (i, cam) in enumerate(self._cams):
                if len(cam) < 1:
                    continue
                # (n_frame, ts, frame) = cam.frame_deque.pop()
                (n_frame, ts, frame) = cam[-1]
                counter += 1
                if starting_time == -1:
                    starting_time = ts

                n, m = frame.shape[0:2]

                #(n_frame, ts) = cam.latest_frame.get_meta()

                if n_frame is None or n_frame == old_id[i]:
                    continue

                #frame = cam.latest_frame.get_frame()

                # fps = 1000000 / (ts - old_timestamp[i] + 0.00001)

                frame8b = frame.copy()  # cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2BGR)
                # frame8b = cv2.resize(frame8b, None, fx=0.5, fy=0.5)
                # frame8b = cv2.resize(frame8b, (256, 256))
                selfie_results = selfie.process(frame8b)
                output_frame = self.segment_func(frame8b)

                landmarks = face_mesh.process(frame8b).multi_face_landmarks
                if landmarks is not None:
                    l = np.array(
                        [[m * lm.x, n * lm.y, lm.z] for lm in landmarks[0].landmark])
                    output_frame = draw_landmarks(frame8b.copy(), l)

                    #p4 = 243
                    #p1 = 130

                    #p2 = 27
                    #p3 = 23

                    p1 = 27
                    p2 = 23

                    p3 = 257
                    p4 = 253

                    def norm(e1, e2):
                        return math.sqrt((e1[0] - e2[0])**2 +
                                         (e1[1] - e2[1])**2 +
                                         (e1[2] - e2[2])**2)

                    l_raw = np.array(
                        [[lm.x, lm.y, lm.z] for lm in landmarks[0].landmark])
                    # 3D euclidean distance:
                    distance = norm(l_raw[p1], l_raw[p4]) + norm(l_raw[p2], l_raw[p3])
                    distance *= 0.5
                    ear_array.append(distance)
                    ts_array.append(ts)

                    self.ear = (np.array(ts_array), np.array(ear_array))
                    if len(ts_array) > 150:
                        ts_array = list(np.array(ts_array)[-150:])
                        ear_array = list(np.array(ear_array)[-150:])

                # cv2.imshow("frame", frame8b)

                if selfie_results.segmentation_mask is not None:
                    mask = selfie_results.segmentation_mask > 0.4

                    # frame8b = segment_image(frame8b, mask)
                    frame8b = self.segment_func(frame8b)

                    mean_rgb = np.sum(frame8b.astype(float), axis=(0, 1)) / np.sum(mask.astype(float), axis=(0, 1))
                    mean_rgb_frames.append(mean_rgb)
                    time_stamps.append(ts - starting_time)

                    if len(mean_rgb_frames) > 100:
                        ts_arr = np.array(time_stamps)
                        # print(ts_arr)
                        time_resampled = np.arange(ts_arr[0], ts_arr[-1], 1000000 / fps, float)
                        idx = np.round(np.interp(time_resampled, ts_arr,
                                       np.arange(0, ts_arr.shape[0]))).astype(int)
                        idx = idx[-100:]

                        start = time.time()
                        signal = _pos_calculation(np.array(mean_rgb_frames)[idx])

                        fourier = np.fft.fft(signal)
                        freq = 60 * np.fft.fftfreq(signal.size, d=1/fps)
                        fourier = fourier[freq >= 40]
                        freq = freq[freq >= 40]
                        fourier = fourier[freq <= 140]
                        freq = freq[freq <= 140]
                        heartrate = freq[np.argmax(fourier)]
                        hr_values.append(heartrate)

                        hr_real_values.append(np.mean(np.array(hr_values)))
                        hr_real_ts.append(ts_arr[-1])

                        self.hr = (ts_arr[idx], signal)
                        self.hr_real = (np.array(hr_real_ts), np.array(hr_real_values))

                        if len(hr_real_ts) > 100:
                            hr_real_ts = list(np.array(hr_real_ts)[-100:])
                            hr_real_values = list(np.array(hr_real_values)[-100:])

                        if len(hr_values) > 300:
                            hr_values = list(np.array(hr_values)[-300:])

                        # freqs, spectrum = _createSpectrum(signal, 5)
                        #print(signal.shape)
                        #print("pos estimation took " + str(time.time() - start))
                        #print("heart rate is " + str(np.mean(np.array(hr_values))))

                        time_stamps = list(ts_arr[-100:])
                        mean_rgb_frames = list(np.array(mean_rgb_frames)[-100:])

                if hr_values:
                    hr_value = hr_values[-1].astype(float)
                    time_stamp = time_stamps[-1].astype(float)
                else:
                    hr_value = 0
                    time_stamp = 0
                if ear_array:
                    ear_value = float(ear_array[-1])
                else:
                    ear_value = 0
                self.new_frame.emit(output_frame,
                                               hr_value,
                                               time_stamp)
                self.change_plot_signal.emit(hr_value, time_stamp)
                self.change_ear_signal.emit(ear_value, time_stamp)
                # cv2.imshow("segmented", frame8b)
                # cv2.waitKey(1)
                #text = "#frame " + str(n_frame) + "\n fps = " + str(fps)
                #cv2.putText(frame8b, text,
                #            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

                # cv2.imshow(str(i), frame8b)
                # cv2.waitKey(1)

                old_id[i] = n_frame
                old_timestamp[i] = ts

            time.sleep(1/300)

    def get_hr(self):
        return self.hr

    def get_hr_real(self):
        return self.hr_real

    def get_ear(self):
        return self.ear


def _pos_calculation(mean_rgb, fps=30):
    win_size = int(fps * 1.6)  # "l" in the paper
    H = np.zeros(mean_rgb.shape[0])
    P_p = np.array([[0, 1, -1], [-2, 1, 1]])  # Projection matrix

    for t in range(0, (mean_rgb.shape[0] - win_size)):
        # Spatial averaging
        C = mean_rgb[t:t + win_size - 1, :].T
        # Temporal normalization
        C_n = np.matmul(np.linalg.inv(np.diag(np.mean(C, axis=1))), C)
        # Projection
        S = np.matmul(P_p, C_n)
        # Alpha Tuning --> [1 alpha] * S
        # alpha = numpy.std(S[0]) / numpy.std(S[1])
        # h = S[0] +  alpha * S[1]
        h = np.matmul(np.array([1, np.std(S[0]) / np.std(S[1])]), S)
        # Overlap adding
        H[t:t + win_size - 1] = H[t:t + win_size - 1] + (h - np.mean(h)) / np.std(h)

    signal = H.flatten()
    return signal