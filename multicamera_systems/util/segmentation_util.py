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


def segment_skin(image):
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
    upper_ycrcb = np.array([255, 173, 127], dtype=np.uint8)
    mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)

    lower_hsv = np.array([0, 30, 0], dtype=np.uint8)
    upper_hsv = np.array([30, 200, 255], dtype=np.uint8)
    mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)

    combined_mask = cv2.bitwise_and(mask_ycrcb, mask_hsv)
    kernel = np.ones((3,3), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

    skin_segmented = cv2.bitwise_and(image, image, mask=combined_mask)
    return skin_segmented
