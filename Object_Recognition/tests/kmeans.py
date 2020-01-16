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
    hog = Hog_descriptor(cv.cvtColor(image, cv.COLOR_RGB2GRAY), cell_size=8, bin_size=8)
    vector, window1 = hog.extract()
    window2 = rgbtohsi(image)[:, :, 0]
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
    estimator = KMeans(n_clusters=2)
    estimator.fit(dataset)
    label_pred = estimator.labels_  # 获取聚类标签
    centroids = estimator.cluster_centers_  # 获取聚类中心
    inertia = estimator.inertia_  # 获取聚类准则的总和
    # with open('prediction.txt', 'w') as f:
    #     f.write(str(label_pred))
    label_pred = label_pred.reshape([64, 76])
    print(label_pred)
    # cv.imshow('rough show', label_pred)


if __name__ == '__main__':
    # cv.namedWindow('input image', cv.WINDOW_NORMAL)
    src = cv.imread('../src.jpg')
    # src = cv.imread('../lena.jpg')
    # cv.imshow('input image', src)
    print("shape of input image: %s" % str(src.shape))
    # print(hog_feature(src))
    # test_hsi_hog(src)

    # cells = split_cells(cv.cvtColor(src, cv.COLOR_RGB2GRAY), 100)
    # n = 0
    # for cell in cells:
    #     print(cell)
    #     print(cell.shape)
    #     cv.imshow('%d image' % n, cell)
    #     n += 1
    #     if n > 10:
    #         break

    dataset = pack_dataset(src, 10)  # 调整小图的边长大小
    print('shape of processed data: %s' % str(dataset.shape))
    #
    bi_means(dataset, )
    cv.waitKey(0)

    cv.destroyAllWindows()
