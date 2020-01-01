import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def dft(img1):
    f = np.fft.fft2(img1)
    fshift = np.fft.fftshift(f)
    rows, cols = img1.shape[:2]
    crow, ccol = np.ceil(rows / 2), np.ceil(cols / 2)
    crow, ccol = np.int(crow), np.int(ccol)
    # 注意fshift是用来与原图像进行掩模操作的但是具体的，我也看着很抽象。这一部分与低通的有些相对的意思。
    fshift[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    plt.subplot(1, 2, 1), plt.imshow(img1)
    plt.title('input image'), plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2), plt.imshow(img_back)
    plt.title('image after HPF'), plt.xticks([]), plt.yticks([])
    plt.show()
    cv.imshow('image fft', img_back)
    return img_back


src = cv.imread("../type1.jpg")
cv.imshow('demo', src)
img = dft(src)

cv.waitKey(0)
