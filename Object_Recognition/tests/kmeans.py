import cv2 as cv
from sklearn.cluster import KMeans
import numpy as np
from collections import Counter
from hog import Hog_descriptor
from hsi import rgbtohsi, calculate
from features import hog_feature
from resize import resize


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


# def pack_dataset(image, n):
#     # h, w = image.shape[:2]
#     # image = cv.resize(image, (h//n*n, w//n*n))
#     hog = Hog_descriptor(cv.cvtColor(image, cv.COLOR_RGB2GRAY), cell_size=8, bin_size=8)
#     vector, window1 = hog.extract()
#     window2 = rgbtohsi(image)[:, :, 2]
#     dataset = []
#     cells_1 = split_cells(window1, n)
#     cells_2 = split_cells(window2, n)
#     for i in range(len(cells_1)):
#         # print(cells_1[i].shape, cells_1[i].dtype)  # (10, 10) uint8
#         # print(cells_2[i].shape, cells_2[i].dtype)  # (10, 10) uint8
#         # TypeError: only integer scalar arrays can be converted to a scalar index
#         array = np.vstack((cells_1[i], cells_2[i])).ravel()
#         dataset.append(array)
#         # print(cells_1[i])
#     return np.array(dataset)

# 采用对H特征进行HOG提取
# def pack_dataset(image, n):
#     # h, w = image.shape[:2]
#     # image = cv.resize(image, (h//n*n, w//n*n))
#     # hog = Hog_descriptor(rgbtohsi(image)[:,:, 0], cell_size=8, bin_size=8)
#     hog = Hog_descriptor(cv.cvtColor(image, cv.COLOR_BGR2GRAY), cell_size=8, bin_size=8)
#     vector, window1 = hog.extract()
#     # window2 = rgbtohsi(image)[:, :, 2]
#     dataset = []
#     cells_1 = split_cells(window1, n)
#     # cells_2 = split_cells(window2, n)
#     for i in range(len(cells_1)):
#         # print(cells_1[i].shape, cells_1[i].dtype)  # (10, 10) uint8
#         # print(cells_2[i].shape, cells_2[i].dtype)  # (10, 10) uint8
#         # TypeError: only integer scalar arrays can be converted to a scalar index
#         # array = np.vstack((cells_1[i])).ravel()
#         array = cells_1[i].ravel()
#         dataset.append(array)
#         # print(cells_1[i])
#     return np.array(dataset)


def pack_dataset(image, n):
    # hog = Hog_descriptor(cv.cvtColor(image, cv.COLOR_RGB2GRAY), cell_size=8, bin_size=8)
    # vector, window1 = hog.extract()
    window2 = rgbtohsi(image)[:, :, 0]
    # window2 = calculate(window2)
    dataset = []
    # cells_1 = split_cells(window1, n)
    cells_2 = split_cells(window2, n)
    for i in range(len(cells_2)):
        # print(cells_1[i].shape, cells_1[i].dtype)  # (10, 10) uint8
        # print(cells_2[i].shape, cells_2[i].dtype)  # (10, 10) uint8
        # TypeError: only integer scalar arrays can be converted to a scalar index
        array = np.array(cells_2[i], np.int8).ravel()
        dataset.append(array)
        # print(cells_1[i])
    return np.array(dataset)


def visualize(arr, image, n):
    """
    根据传入的image和n，resize arr数组为2D适配image规格，并据此渲染原图
    :param arr:
    :param image:
    :param n:
    :return:
    """
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    h, w = gray.shape[:2]
    h //= 10
    w //= 10
    # image = cv.resize(image, (h,w))
    # FIX A BUG BY RANDOM ALLOCATION OF 2_MEANS CLASSIFIER
    counter = Counter(arr)
    which2render = 1 if counter[0]>counter[1] else 0
    arr = np.reshape(arr, (h, w))
    for i in range(h):
        for j in range(w):
            # 对焦点区域进行颜色标记
            if arr[i, j] == which2render:
                # gray[i*N:(i+1)*N, j*N:(j+1)*N] = 100
                image[i*n:(i+1)*n, j*n:(j+1)*n, 2] = 255
    # cv.imshow("after render", image)
    # cv.imshow("after render", cv.cvtColor(gray, cv.COLOR_GRAY2BGR))
    cv.imshow("after render", image)


def test_hsi_hog(image):
    """
    用于演示hsi特征提取效果
    height, width = self.img.shape
    ValueError: too many values to unpack (expected 2)
    :param image:
    :return:
    """
    img = rgbtohsi(image)
    vector, img = Hog_descriptor(img, cell_size=8, bin_size=8).extract()
    cv.imshow('hog after hsi', img)
    print(np.array(vector).shape)


def bi_means(dataset):
    """
    二分K平均模型
    :param dataset:
    :return: 1D array
    """
    estimator = KMeans(n_clusters=2)
    estimator.fit(dataset)
    label_pred = estimator.labels_  # 获取聚类标签
    centroids = estimator.cluster_centers_  # 获取聚类中心
    inertia = estimator.inertia_  # 获取聚类准则的总和

    # with open('prediction.txt', 'w') as f:
    #     f.write(str(label_pred))
    print(label_pred)
    return label_pred


def main(src):
    cv.namedWindow('input image', cv.WINDOW_NORMAL)
    N = 10  # 小方格像素值
    src = resize(src, N)
    cv.imshow('input image', src)
    print("shape of input image: %s" % str(src.shape))
    dataset = pack_dataset(src, N)
    print('shape of processed data: %s' % str(dataset.shape))
    labels = bi_means(dataset)
    visualize(labels, src, N)
    cv.waitKey(0)
    cv.destroyAllWindows()


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
    main(src)
