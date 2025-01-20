import time
from PyQt6.QtCore import QThread
from collections import deque


class MultimodalWorker(QThread):
    def __init__(self):
        QThread.__init__(self)
        self._cams = []

    def add_cam(self, cam):
        frame_deque = deque(maxlen=10)
        self._cams.append(frame_deque)
        cam.new_frame.connect(
            lambda frame, n_frame, ts: frame_deque.append((n_frame, ts, frame)))


class TriggerThread(MultimodalWorker):
    def __init__(self, fps=30):
        MultimodalWorker.__init__(self)
        self.__fps = fps

    def run(self):
        while True:
            start = time.time()
            try:
                for c in self._cams:
                    c.cam.set_trigger_software(1)
            except:
                pass
            elapsed = time.time() - start
            sleep_time = (1 / self.__fps) - elapsed
            time.sleep(max(sleep_time, 0))


class DataIOThread(MultimodalWorker):
    def __init__(self):
        MultimodalWorker.__init__(self)

    def run(self):
        # example thread that can later do the hdf5 writing (pull left from the deque),
        # right now it prints the current, effective framerate
        old_timestamp = [-1] * len(self._cams)
        old_id = [-1] * len(self._cams)

        while True:
            for (i, cam) in enumerate(self._cams):
                if len(cam) > 1:
                    (n_frame, ts, frame) = cam.popleft()

                    # if n_frame != old_id[i] + 1 and old_id[i] != -1:
                    #    print("Frame dropped!")

                    try:
                        fps = 1000000 / (ts - old_timestamp[i])
                        text = "Framerate: " + str(fps) + "\n"
                        # print(text)
                    except:
                        pass

                    old_id[i] = n_frame
                    old_timestamp[i] = ts
                    del(frame)
            time.sleep(0.001)