import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def plot_demo(image):
    """
    numpy中的ravel()、flatten()、squeeze()都有将多维数组转换为一维数组的功能，区别：
    ravel()：如果没有必要，不会产生源数据的副本
    flatten()：返回源数据的副本
    squeeze()：只能对维数为1的维度降维
    :param image: cv2.imread返回的对象
    :return:
    """
    plt.hist(image.ravel(), bins=256, range=[0, 255])
    plt.show()


def image_hist(image):
    """
    def calcHist(images: Any,
             channels: Any,
             mask: Any,
             histSize: Any, -> [256] for example
             ranges: Any,
             hist: Any = None,
             accumulate: Any = None) -> None
    :param image:
    :return:
    """
    colors = ('blue', 'green', 'red')
    for i, color in enumerate(colors):
        hist = cv.calcHist([image], [i], None, [256], [0, 255])
        plt.plot(hist, color=color)
        plt.xlim(0, 256)
    plt.show()


def equal_hist_demo(image):
    """
    直方图全局均衡化
    注意：均衡化只认识灰度图
    :param image:
    :return:
    """
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    dst = cv.equalizeHist(gray)
    cv.imshow("equal hist demo", dst)


def clahe_demo(image):
    """
    直方图局部均衡化
    优点在于可以人为干涉
    :param image:
    :return:
    """
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    clahe = cv.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    dst = clahe.apply(gray)
    # cv.imshow("clahe demo", dst)
    cv.imshow("clahe demo", cv.cvtColor(dst, cv.COLOR_GRAY2BGR))


def clahe_rgb(image):
    """
    测试：有去雾效果
    :param image:
    :return:
    """
    h, w, c = image.shape
    split_bgr = cv.split(image)
    for i in range(c):
        clahe = cv.createCLAHE(clipLimit=4)
        split_bgr[i] = clahe.apply(split_bgr[i])
    cv.merge(split_bgr, image)
    cv.imshow("clahe rgb", image)


def create_rgb_hist(image):
    """
    创建rgb直方图
    :param image:
    :return:
    """
    h, w, c = image.shape
    # 直方图降维操作
    rgb_hist = np.zeros([16*16*16, 1], np.float32)
    # 相应的bins变为16
    bsize = 256 / 16
    for row in range(h):
        for col in range(w):
            b = image[row, col, 0]
            g = image[row, col, 1]
            r = image[row, col, 2]
            index = np.int(b/bsize)*16*16 + np.int(g/bsize)*16 + np.int(r/bsize)
            rgb_hist[np.int(index), 0] = rgb_hist[np.int(index), 0] + 1
    return rgb_hist


def hist_compare(image1, image2):
    """
    直方图之间的比较，使用巴氏距离、相关性、卡方
    :param image1:
    :param image2:
    :return:
    """
    hist1 = create_rgb_hist(image1)
    hist2 = create_rgb_hist(image2)
    match1 = cv.compareHist(hist1, hist2, cv.HISTCMP_BHATTACHARYYA)
    match2 = cv.compareHist(hist1, hist2, cv.HISTCMP_CORREL)
    match3 = cv.compareHist(hist1, hist2, cv.HISTCMP_CHISQR)
    print("巴氏距离:%s, 相关性:%s, 卡方:%s" % (match1, match2, match3))


if __name__ == '__main__':
    print("OpenCV demo")
    # src = cv.imread('../type2.jpg')
    # src = cv.imread('../type5.jpg')
    # src = cv.imread('../type6.jpg')
    # src = cv.imread('../lifeboat.jpg')
    src = cv.imread('../vague.png')
    cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
    cv.imshow("input image", src)
    # plot_demo(src)
    # image_hist(src)
    # equal_hist_demo(src)
    # clahe_demo(src)
    clahe_rgb(src)
    # src1 = cv.imread("../type1.jpg")
    # src2 = cv.imread("../type2.jpg")
    # src2 = cv.imread("../type1.jpg")
    # hist_compare(src1, src2)
    cv.waitKey(0)

    cv.destroyAllWindows()
