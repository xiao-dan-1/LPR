import os

import cv2
import easyocr
import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']


def plt_show(name, img):
    plt.imshow(img)
    plt.title(name)
    plt.show()


img_path = r"D:\Desktop\license plate recognition\My_LPR\datasets\CCPD\lp"
img_name = "云A005HL.jpg"
# image_names = os.listdir(img_path)
# # img_name = np.random.choice(image_names, 1)[0]
#
# # print(image_names)
#
image_path = os.path.join(img_path, img_name)

reader = easyocr.Reader(['ch_sim', 'en'])  # this needs to run only once to load the model into memory
image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
results = reader.readtext(image)

for result in results:
    pts, string, acc = result
    print(pts, string, acc)
    cv2.polylines(image, np.array([pts]), 1, 255)
#
cv2.imshow('image', image)
cv2.waitKey()
