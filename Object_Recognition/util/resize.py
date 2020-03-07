import win32api, win32con
import cv2


def resize(image, n):
    """
    获取windows系统的屏幕尺寸
    :param image:
    :param n:
    :return:
    """
    h, w = image.shape[:2]
    x = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
    y = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
    # 最大占2/3屏幕
    x = x * 2 // 3
    y = y * 2 // 3
    # 按照原图比例调节
    if h > x or w > y:
        k = min(x/h, y/w)
        h = int(h * k)
        w = int(w * k)
    h_bins = h // n
    w_bins = w // n
    return cv2.resize(image, (w_bins*n, h_bins*n), interpolation=cv2.INTER_AREA)
