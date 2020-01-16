from sklearn.cluster import KMeans
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def simple_classification(image):
    image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    img = image.ravel()
    img.resize(len(img), 1)  # 化为n*1二维数组
    k_means = KMeans(n_clusters=2)
    k_means.fit(img)
    pre = k_means.predict(img)
    pre.resize(512, 512)
    pre = pre ^ np.ones([512, 512], np.uint8)
    plt.imshow(pre, cmap=plt.cm.gray)
    plt.show()
    # cv.imshow('binary classification of gray', pre)


if __name__ == '__main__':
    src = cv.imread('../lena.jpg')
    cv.namedWindow('input image', 1)
    cv.imshow('input image', src)
    simple_classification(src)
    cv.waitKey(0)
    cv.destroyAllWindows()
