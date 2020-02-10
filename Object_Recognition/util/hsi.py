import cv2 as cv
import numpy as np
np.seterr(divide='ignore',invalid='ignore')

def rgbtohsi(rgb_lwp_img):
    """
    解决数组运算出现RuntimeWarning: invalid value encountered
    http://blog.sciencenet.cn/blog-3236941-1048188.html
    :param rgb_lwp_img:
    :return:
    """
    rows = int(rgb_lwp_img.shape[0])
    cols = int(rgb_lwp_img.shape[1])
    b, g, r = cv.split(rgb_lwp_img)
    # 归一化到[0,1]
    b = b / 255.0
    g = g / 255.0
    r = r / 255.0
    hsi_lwp_img = rgb_lwp_img.copy()
    H, S, I = cv.split(hsi_lwp_img)
    for i in range(rows):
        for j in range(cols):
            num = 0.5 * ((r[i, j]-g[i, j])+(r[i, j]-b[i, j]))
            den = np.sqrt((r[i, j]-g[i, j])**2+(r[i, j]-b[i, j])*(g[i, j]-b[i, j]))
            theta = float(np.arccos(num/den))

            if den == 0:
                    H = 0
            elif b[i, j] <= g[i, j]:
                H = theta
            else:
                H = 2*3.14169265 - theta

            min_RGB = min(min(b[i, j], g[i, j]), r[i, j])
            sum = b[i, j]+g[i, j]+r[i, j]
            if sum == 0:
                S = 0
            else:
                S = 1 - 3*min_RGB/sum
            # 这一步对H要归一化到(0, 1)
            H = H/(2*3.14159265)
            I = sum/3.0
            # 输出HSI图像，扩充到255以方便显示，一般H分量在[0,2pi]之间，S和I在[0,1]之间
            hsi_lwp_img[i, j, 0] = H*255
            hsi_lwp_img[i, j, 1] = S*255
            hsi_lwp_img[i, j, 2] = I*255
    # cv.imshow('hsi', hsi_lwp_img)
    # return hsi_lwp_img[:,:,0]
    return calculate(hsi_lwp_img[:,:,0])


def calculate(np_arr):
    arr = list(np_arr.ravel())
    from collections import Counter
    counter = Counter(arr)
    # from pprint import pprint
    # pprint(counter)
    for key, value in counter.items():
        # 绝对参数：1000, 5000 相对参数：0.008 待调参数
        if value > 0.008*len(arr):
            value = 255
        else:
            value = 0
        np_arr[np_arr == key] = value
    ret, binary = cv.threshold(np_arr, 0, 255, cv.THRESH_BINARY | cv.THRESH_TRIANGLE)
    # cv.imshow("binary", binary)
    se = cv.getStructuringElement(cv.MORPH_RECT, (3, 3), (-1, -1))
    # 闭操作
    # dilate = cv.dilate(binary, se, None, (-1, -1), 1)
    # erode = cv.erode(dilate, se, None, (-1, -1), 1)
    # 开操作
    erode = cv.erode(binary, se, None, (-1, -1), 1)
    dilate = cv.dilate(erode, se, None, (-1, -1), 1)

    # 显示
    # cv.imshow("erode", erode)
    # cv.imshow("dilate", dilate)
    return dilate
