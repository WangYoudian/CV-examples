import os
from pipeline import pipeline

IMAGE_DIR = 'images/'
roi = 'roi.png'


for image in os.listdir(IMAGE_DIR):
    pipeline(os.path.join(IMAGE_DIR, image), roi, image)
    # print(image)
