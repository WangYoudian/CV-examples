import time
import cv2 as cv
from sklearn.cluster import KMeans
import numpy as np
from collections import Counter
from util.hog import Hog_descriptor
from util.hsi import rgbtohsi
from util.resize import resize
from util.connected_area import *


def split_cells(image, n, dy=10, dx=10):
    """
    将图片切割成长宽为dx和dy的小矩形图片
    :param image: np.array 2D
    :param n: cell size of pixel(s)
    :return:
    """
    cells = []
    dx = dy = n
    h, w = image.shape[:2]
    x1 = 0
    y1 = 0
    x2 = dx
    y2 = dy
    while x2 <= h:
        while y2 <= w:
            cell = np.array(image[x1:x2, y1:y2], np.uint8)
            # if cells is None:
            #     cells = cell
            # else:
            #     cells = np.vstack((cells, cell))
            cells.append(cell)
            y1 = y1 + dy
            y2 = y2 + dy
        x1 = x1 + dx
        x2 = x2 + dx
        y1 = 0
        y2 = dy
    return cells


def pack_dataset(image, n):
    # hog = Hog_descriptor(cv.cvtColor(image, cv.COLOR_RGB2GRAY), cell_size=8, bin_size=8)
    # vector, window1 = hog.extract()
    # TODO:5s
    window2 = rgbtohsi(image)
    dataset = []
    # TODO:0.002s
    cells_2 = split_cells(window2, n)
    for i in range(len(cells_2)):
        array = np.array(cells_2[i], np.int8).ravel()
        dataset.append(array)
    return np.array(dataset)


def connected_areas(shape, arr, n):
    """
    :param shape: [h, w]
    :param arr: 二分类的预测数据
    :param n: 小正方形边长
    :return: 包含连通域信息的二维数组
    """
    h, w = shape
    h //= n
    w //= n
    counter = Counter(arr)
    # 防止二分类随机性造成的标记错误（0标为1,1标为0）
    minor = 1 if counter[0] > counter[1] else 0
    arr = np.reshape(arr, (h, w))
    if minor == 0:
        # 将0换成1，将1换成0
        arr = 1 - arr
    # 变为二值图像数据形式
    arr[arr==minor] = 255
    binary_img = arr
    binary_img = Seed_Filling(binary_img, NEIGHBOR_HOODS_8)
    binary_img, points = reorganize(binary_img)
    # print(binary_img, points)
    # print(len(points))
    result = []
    for area in points:
        x1, y1, x2, y2 = h, w, 0, 0
        if len(area) < 3:
            continue
        # 提取最左上和右下的坐标
        for point in area:
            x1 = point[0] if point[0] < x1 else x1
            x2 = point[0] if point[0] > x2 else x2
            y1 = point[1] if point[1] < y1 else y1
            y2 = point[1] if point[1] > y2 else y2
        # 还原为原图的比例
        result.append([x1*n, y1*n, x2*n, y2*n, 1])
    return np.array(result)


def bi_means(dataset):
    """
    二分K平均模型
    :param dataset:
    :return: 1D array 表示将原图从右到左 从上到下分割每个小块的标记
    """
    estimator = KMeans(n_clusters=2)
    estimator.fit(dataset)
    label_pred = estimator.labels_  # 获取聚类标签
    return label_pred


def process(src):
    """
    接收cv.imread对象，返回连通域的数组
    :param src:
    :return:
    """
    N = 20  # 小方格像素值
    src = resize(src, N)
    # cv.imshow('input image', src)
    # print("shape of input image: %s" % str(src.shape))
    # TODO:5s
    dataset = pack_dataset(src, N)
    # print('shape of processed data: %s' % str(dataset.shape))
    # TODO：0.13s
    labels = bi_means(dataset)
    # TODO：0.004s
    res = connected_areas(src.shape[:2], labels, N)
    print('Interesting area:%s' % str(res))
    return res


if __name__ == '__main__':
    # manually
    start = cv.getTickCount()
    path = 'images/26.jpg'
    # path = ('images/1.jpg')
    src = cv.imread(path)
    end = cv.getTickCount()
    print("time costs:%.3f s" % ((end-start)/cv.getTickFrequency()))
    ans = process(src)

