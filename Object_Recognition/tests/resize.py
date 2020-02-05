import win32api, win32con
import cv2


def resize(image, n):
    h, w = image.shape[:2]
    x = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
    y = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
    x = x * 2 // 3
    y = y * 2 // 3
    if h > x or w > y:
        k = min(x/h, y/w)
        h = int(h * k)
        w = int(w * k)
    h_bins = h // n
    w_bins = w // n
    return cv2.resize(image, (w_bins*n, h_bins*n), interpolation=cv2.INTER_AREA)
