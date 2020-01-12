import cv2 as cv
from sklearn.cluster import KMeans
import numpy as np
from tests.hog import Hog_descriptor
from tests.hsi import rgbtohsi
from tests.features import hog_feature

def test_hsi_hog(image):
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
    return 0


if __name__ == '__main__':
    cv.namedWindow('input image', cv.WINDOW_NORMAL)
    src = cv.imread('../src.jpg')
    cv.imshow('input image', src)
    # print(hog_feature(src))
    test_hsi_hog(src)
    cv.waitKey(0)

    cv.destroyAllWindows()
