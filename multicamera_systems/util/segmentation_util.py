import mediapipe as mp
import numpy as np
import cv2


class Segmenter:
    def __init__(self):
        self.mp_face_segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0)

    def __call__(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.mp_face_segmentation.process(image)
        mask = results.segmentation_mask
        mask = np.stack((mask,) * 3, axis=-1)
        return mask


def segment_image(img, mask):
    img[np.invert(np.repeat(np.expand_dims(mask, 2), 3, axis=2))] = 0
    return img
