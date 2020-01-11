import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


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
    # cv.imshow("back projection", dst)

    # 将分散的点连接
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    dst = cv.filter2D(dst, -1, kernel=kernel)
    # 二值化
    ret, tr = cv.threshold(dst, 50, 255, 0)
    # 把tr变为三通道
    tr = cv.merge((tr, tr, tr))
    # 把tr作为蒙板，在原图上作用
    res = cv.bitwise_and(target, tr)
    # cv.imshow("target picture", target)
    # cv.imshow("sectional draw", res)
    return res


def edge_demo(image):
    """
    Canny边缘检测
    :param image:
    :return: gray
    """
    blurred = cv.GaussianBlur(image, (3, 3), 0)
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    # cv.imshow("gray", gray)
    # clahe = cv.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    # dst = clahe.apply(gray)
    dst = cv.equalizeHist(gray)
    # cv.imshow("equalize by hist", dst)

    # grad_x = cv.Sobel(gray, cv.CV_16SC1, 1, 0)
    # grad_y = cv.Sobel(gray, cv.CV_16SC1, 0, 1)
    # edge_output = cv.Canny(grad_x, grad_y, 30, 150)

    edge_output = cv.Canny(dst, 50, 200)
    # cv.imshow("Canny demo", edge_output)
    return edge_output


def measure_object(image, source, path):
    """
    画轨迹
    :param image: gray
    :param source: bgr source picture
    :return:
    """
    # gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(image, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    print("threshold value: %s"%ret)
    # cv.imshow("binary image", binary)

    contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours):
        # cv.drawContours(source, contours, i, (0, 255, 255), 1)  # 用黄色线条画出轮廓

        area = cv.contourArea(contour)  # 计算轮廓面积
        # if area > 300:
        if area > 30:
            print("contour area:", area)

            # 轮廓周长,第二参数可以用来指定对象的形状是闭合的（True）,还是打开的（一条曲线）。
            perimeter = cv.arcLength(contour, True)
            print("contour perimeter:", perimeter)

            x, y, w, h = cv.boundingRect(contour)  # 用矩形框出轮廓
            # rate = min(w, h)/max(w, h)  # 计算矩形宽高比
            rate = max(h/source.shape[0], w/source.shape[1])
            if rate > 0.05:
                # if rate:
                print("rectangle rate", rate)
                cv.rectangle(source, (x, y), (x + w, y + h), (0, 0, 255), 2)
                mm = cv.moments(contour)  # 函数 cv2.moments() 会将计算得到的矩以一个字典的形式返回
                # 计算出对象的重心
                try:
                    cx = mm['m10']/mm['m00']
                    cy = mm['m01']/mm['m00']
                except ZeroDivisionError:
                    cx = 0
                    cy = 0
                cv.circle(source, (np.int(cx), np.int(cy)), 2, (0, 255, 255), -1)  # 用实心圆画出重心

    cv.imshow("measure_object", source)
    cv.imwrite("result/"+path, source)


def pipeline(image, *args):
    cv.namedWindow('input image', cv.WINDOW_AUTOSIZE)
    src = cv.imread(image)
    cv.imshow("input image", src)
    if len(args) >= 1:
        roi = cv.imread(args[0])
        res = sectional_draw(roi, src)
        enhance = edge_demo(res)
        measure_object(enhance, src, args[1])
    else:
        pass

    # img = cv.imread("../images/approximate.png")
    # img = cv.imread("../images/lifeboat.jpg")
    # img = cv.imread("../images/helicopter.jpg")
    # contour_approx(img)
    cv.waitKey(0)  # 等有键输入或者1000ms后自动将窗口消除，0表示只用键输入结束窗口
    cv.destroyAllWindows()  # 关闭所有窗口


if __name__ == '__main__':
    roi = 'roi.png'
    path = 'type6.jpg'
    # path = 'helicopter.jpg'
    # src = cv.imread("helicopter.jpg")
    # roi = cv.imread('roi.png')
    # src = cv.imread("type1.jpg")
    # src = cv.imread("type2.jpg")
    # src = cv.imread("type3.jpg")
    # src = cv.imread("type4.jpg")
    # src = cv.imread("type5.jpg")
    # src = cv.imread("type6.jpg")
    # src = cv.imread(path)
    src = path
    pipeline(src, roi, path)
