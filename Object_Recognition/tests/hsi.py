import cv2 as cv
import numpy as np
from resize import resize
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
    return hsi_lwp_img


def calculate(np_arr):
    arr = list(np_arr.ravel())
    from collections import Counter
    counter = Counter(arr)
    from pprint import pprint
    pprint(counter)
    for key, value in counter.items():
        if value > 1000:
            value = 255
        else:
            value = 0
        np_arr[np_arr == key] = value
    ret, binary = cv.threshold(np_arr, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imshow("binary", binary)
    se = cv.getStructuringElement(cv.MORPH_RECT, (3, 3), (-1, -1))
    dilate = cv.dilate(binary, se, None, (-1, -1), 1)
    erode = cv.erode(dilate, se, None, (-1, -1), 1)

    # 显示
    cv.imshow("erode", erode)
    cv.imshow("dilate", dilate)
    return dilate


if __name__ == '__main__':
    # FOR TEST
    # src = cv.imread('../src.jpg')
    # src = cv.imread('../background/1.jpg')
    # src = cv.imread('../background/2.jpg')
    # src = cv.imread('../background/10.jpg')
    # src = cv.imread('../background/4.jpg')
    # src = cv.imread('../background/5.jpg')
    # src = cv.imread('../background/6.jpg')
    # src = cv.imread('../background/7.jpg')
    # src = cv.imread('../background/8.jpg')
    # src = cv.imread('../background/9.jpg')
    # src = cv.imread('../background.jpg')

    # FOR EXPERIMENT
    # src = cv.imread('../images/2.jpg')
    src = cv.imread('../images/26.jpg')
    src = resize(src, 10)
    start = cv.getTickCount()
    hsi = rgbtohsi(src)
    end = cv.getTickCount()
    print('Function costs ' + str((end - start)/cv.getTickFrequency()) + ' second(s)')
    cv.imshow('rgb_lwpImg', src)
    cv.imshow('hsi_lwpImg', hsi[:,:,0])
    print(hsi[:,:,0].max(), hsi[:,:,0].min())
    H = hsi[:,:,0]
    calculate(H)
    cv.imwrite('rgb_lwpImg.jpg', src)
    cv.imwrite('hsi_lwpImg.jpg', hsi)
    cv.imwrite('hsi_lwpImg0.jpg', hsi[:,:,0])
    print('hsi shape:' + str(hsi.shape))
    key = cv.waitKey(0)
    if key == ord('q'):
        cv.destroyAllWindows()
