import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def measure_object(image, source):
    """
    形态测量
    :param image: gray
    :param source: bgr source picture
    :return:
    """
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    print("threshold value: %s"%ret)
    cv.imshow("binary image", binary)

    contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours):
        cv.drawContours(image, contours, i, (0, 255, 255), 1)  # 用黄色线条画出轮廓

        area = cv.contourArea(contour)  # 计算轮廓面积
        print("contour area:", area)

        # 轮廓周长,第二参数可以用来指定对象的形状是闭合的（True）,还是打开的（一条曲线）。
        perimeter = cv.arcLength(contour, True)
        print("contour perimeter:", perimeter)
        x, y, w, h = cv.boundingRect(contour)  # 用矩阵框出轮廓
        cv.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        rate = min(w, h)/max(w, h)  # 计算矩阵宽高比
        print("rectangle rate",rate)

        mm = cv.moments(contour)  # 函数 cv2.moments() 会将计算得到的矩以一个字典的形式返回
        # 计算出对象的重心
        try:
            cx = mm['m10']/mm['m00']
            cy = mm['m01']/mm['m00']
        except Exception:
            cx = 0
            cy = 0
        cv.circle(image, (np.int(cx), np.int(cy)), 2, (0, 255, 255), -1)  # 用实心圆画出重心
    cv.imshow("measure_object", image)


def contour_approx(image):
    """
    提取近似轮廓
    :param image:
    :return:
    """
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    print("threshold value: %s" % ret)
    cv.imshow("binary image", binary)

    contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours):
        cv.drawContours(image, contours, i, (0, 0, 255), 2)  # 用红色线条画出轮廓

        # 将轮廓形状近似到另外一种由更少点组成的轮廓形状，新轮廓的点的数目由我们设定的准确度来决定。
        # 为了帮助理解，假设从一幅图像中查找一个矩形，但是由于图像的种种原因，我们不能得到一个完美的矩形，
        # 而是一个“坏形状”（如下图所示）。
        # 现在你就可以使用这个函数来近似这个形状（）了。
        # 这个函数的第二个参数叫 epsilon，它是从原始轮廓到近似轮廓的最大距离。
        # 它是一个准确度参数。选 择一个好的 epsilon 对于得到满意结果非常重要。

        epsilon = 0.01 * cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, epsilon, True)
        cv.drawContours(image, approx, -1, (255, 0, 0), 10)

    cv.imshow("contour_approx", image)