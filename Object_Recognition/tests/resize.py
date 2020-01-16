import cv2


def resize(image, n):
    h, w = image.shape[:2]
    h_bins = h // n
    w_bins = w // n
    return cv2.resize(image, (w_bins*n, h_bins*n), interpolation=cv2.INTER_AREA)
