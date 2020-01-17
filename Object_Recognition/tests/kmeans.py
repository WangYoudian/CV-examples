import cv2 as cv
from sklearn.cluster import KMeans
import numpy as np
from hog import Hog_descriptor
from hsi import rgbtohsi
from features import hog_feature


def split_cells(image, n, dy=10, dx=10):
    """

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
    while x2 < h:
        while y2 < w:
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
    # h, w = image.shape[:2]
    # image = cv.resize(image, (h//n*n, w//n*n))
    hog = Hog_descriptor(cv.cvtColor(image, cv.COLOR_RGB2GRAY), cell_size=8, bin_size=8)
    vector, window1 = hog.extract()
    window2 = rgbtohsi(image)[:, :, 2]
    dataset = []
    cells_1 = split_cells(window1, n)
    cells_2 = split_cells(window2, n)
    for i in range(len(cells_1)):
        # print(cells_1[i].shape, cells_1[i].dtype)  # (10, 10) uint8
        # print(cells_2[i].shape, cells_2[i].dtype)  # (10, 10) uint8
        # TypeError: only integer scalar arrays can be converted to a scalar index
        array = np.vstack((cells_1[i], cells_2[i])).ravel()
        dataset.append(array)
        # print(cells_1[i])
    return np.array(dataset)


def visualize(arr, image, n):
    """
    根据传入的image和n，resize arr数组为2D适配image规格
    :param arr:
    :param image:
    :param n:
    :return:
    """
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    h, w = gray.shape[:2]
    h //= 10
    w //= 10
    image = cv.resize(image, (h,w))
    arr = np.reshape(arr, (h, w))
    for i in range(h):
        for j in range(w):
            # 对焦点区域进行颜色标记
            if arr[i, j] == 1:
                gray[i*N:(i+1)*N, j*N:(j+1)*N] = 100
    # cv.imshow("after render", image)
    cv.imshow("after render", cv.cvtColor(gray, cv.COLOR_GRAY2BGR))


def test_hsi_hog(image):
    """
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
    # label_pred = label_pred.reshape([64, 76])
    # with open('prediction.txt', 'w') as f:
    #     f.write(str(label_pred))
    print(label_pred)
    # cv.imshow('rough show', label_pred)
    return label_pred


if __name__ == '__main__':
    cv.namedWindow('input image', cv.WINDOW_AUTOSIZE)
    src = cv.imread('../src.jpg')
    # src = cv.imread('../lena.jpg')
    # src = cv.imread('../images/1.jpg')
    N = 10  # 小方格像素值
    cv.imshow('input image', src)
    print("shape of input image: %s" % str(src.shape))
    # print(hog_feature(src))
    # test_hsi_hog(src)

    # cells = split_cells(cv.cvtColor(src, cv.COLOR_RGB2GRAY), N)
    # n = 0
    # for cell in cells:
    #     print(cell)
    #     print(cell.shape)
    #     cv.imshow('%d image' % n, cell)
    #     n += 1
    #     if n > 10:
    #         break

    dataset = pack_dataset(src, N)  # 调整小图的边长大小
    print('shape of processed data: %s' % str(dataset.shape))
    #
    labels = bi_means(dataset)
    visualize(labels, src, N)
    cv.waitKey(0)

    cv.destroyAllWindows()
