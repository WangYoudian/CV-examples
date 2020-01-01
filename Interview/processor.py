import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def edge_demo(image):
    """
    Canny边缘检测
    :param image:
    :return: gray
    """
    blurred = cv.GaussianBlur(image, (3, 3), 0)
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    # cv.imshow("gray", gray)

    # grad_x = cv.Sobel(gray, cv.CV_16SC1, 1, 0)
    # grad_y = cv.Sobel(gray, cv.CV_16SC1, 0, 1)
    # edge_output = cv.Canny(grad_x, grad_y, 30, 150)

    edge_output = cv.Canny(gray, 50, 200)
    cv.imshow("Canny demo", edge_output)
    return edge_output


def measure_object(image, source):
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
        cv.drawContours(source, contours, i, (0, 255, 255), 1)  # 用黄色线条画出轮廓

        area = cv.contourArea(contour)  # 计算轮廓面积
        if area > 30:
            print("contour area:", area)

            # 轮廓周长,第二参数可以用来指定对象的形状是闭合的（True）,还是打开的（一条曲线）。
            perimeter = cv.arcLength(contour, True)
            print("contour perimeter:", perimeter)

            x, y, w, h = cv.boundingRect(contour)  # 用矩阵框出轮廓
            rate = min(w, h)/max(w, h)  # 计算矩阵宽高比
            if rate <= 2 and rate > 0.5:
            # if rate:
                print("rectangle rate",rate)

                cv.rectangle(source, (x, y), (x + w, y + h), (0, 0, 255), 2)

                mm = cv.moments(contour)  # 函数 cv2.moments() 会将计算得到的矩以一个字典的形式返回
                # 计算出对象的重心
                try:
                    cx = mm['m10']/mm['m00']
                    cy = mm['m01']/mm['m00']
                except Exception:
                    cx = 0
                    cy = 0
                cv.circle(source, (np.int(cx), np.int(cy)), 2, (0, 255, 255), -1)  # 用实心圆画出重心

    cv.imshow("measure_object", source)


# 近似轮廓
def contour_approx(image):
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
    cv.waitKey(0)
    return img_back


def main():
    # src = cv.imread("helicopter.jpg")
    # src = cv.imread("type1.jpg")
    # src = cv.imread("type2.jpg")
    # src = cv.imread("type3.jpg")
    src = cv.imread("../type4.jpg")
    # src = cv.imread("type5.jpg")
    # src = cv.imread("type6.jpg")
    cv.imshow("demo", src)
    # src = dft(src)
    enhance = edge_demo(src)

    measure_object(enhance, src)

    # img = cv.imread("../images/approximate.png")
    # img = cv.imread("../images/lifeboat.jpg")
    # img = cv.imread("../images/helicopter.jpg")
    # contour_approx(img)
    cv.waitKey(0)  # 等有键输入或者1000ms后自动将窗口消除，0表示只用键输入结束窗口
    cv.destroyAllWindows()  # 关闭所有窗口


if __name__ == '__main__':
    main()
