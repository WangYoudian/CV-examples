import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def sectional_draw(sample, target):
    """
    这其实算是一种自动抠图的方法
    :param sample:
    :param target:
    :return:
    """
    roi_hsv = cv.cvtColor(sample, cv.COLOR_BGR2HSV)
    roi_hist = cv.calcHist(roi_hsv, [0, 1], None, [36, 48], [0, 180, 0, 256])
    cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)
    dst = cv.calcBackProject([target], [0, 1], roi_hist, [0, 180, 0, 256], 1)
    cv.imshow("back projection", dst)

    # 将分散的点连接
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    dst = cv.filter2D(dst, -1, kernel=kernel)
    # 二值化
    ret, tr = cv.threshold(dst, 50, 255, 0)
    # 把tr变为三通道
    tr = cv.merge((tr, tr, tr))
    # 把tr作为蒙板，在原图上作用
    res = cv.bitwise_and(target, tr)
    cv.imshow("target picture", target)
    cv.imshow("sectional draw", res)
    # 对target/tr/res进行手动编排，迭代显示
    s = np.row_stack((target, tr, res))
    cv.imshow("together shown", s)


def back_projection_demo(sample, target):
    roi_hsv = cv.cvtColor(sample, cv.COLOR_BGR2HSV)
    # 在原图上投影，用不上原图hsv
    # target_hsv = cv.cvtColor(target, cv.COLOR_BGR2HSV)

    cv.imshow("sample", sample)
    cv.imshow("target", target)

    roi_hist = cv.calcHist(roi_hsv, [0, 1], None, [180, 256], [0, 180, 0, 256])
    cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)
    dst = cv.calcBackProject([target], [0, 1], roi_hist, [0, 180, 0, 256], 1)
    cv.imshow("back projection demo", dst)


def hist2d_demo(image):
    # hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    hist = cv.calcHist([image], [0, 1], None, [30, 48], [0, 180, 0, 256])
    plt.imshow(hist, interpolation='nearest')
    plt.title('2D Histogram')
    plt.show()


if __name__ == '__main__':
    # cv.namedWindow('input image', cv.WINDOW_AUTOSIZE)
    # src = cv.imread("messi5.jpg")
    # cv.imshow("input image", src)
    # hist2d_demo(src)
    src1 = cv.imread("../roi.png")
    # src2 = cv.imread("target.png")
    src2 = cv.imread("../type6.jpg")
    # back_projection_demo(sample, target)
    sectional_draw(src1, src2)
    cv.waitKey(0)
    cv.destroyAllWindows()
