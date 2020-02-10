import os
import cv2
from kmeans import main

DIR = '../images/'

for file in os.listdir(DIR):
    path = os.path.join(DIR, file)
    main(path)
    cv2.waitKey(0)
