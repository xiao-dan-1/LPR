import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from Progressbar import Progressbar

from CCPD import CCPD

if __name__ == '__main__':
    ccpd_path = "D:\Desktop\license plate recognition\CCPD\CCPD2019"
    train_path = os.path.join(ccpd_path, r"splits\train.txt")
    save_path = "D:\Desktop\license plate recognition\CCPD\CCPD2019\mask"

    files_path = []
    with open(train_path, 'r') as f:
        for line in f.readlines():
            files_path.append(line.replace('\n', ''))
    show_num = 5
    for file_path in Progressbar(files_path):
        file_path = os.path.join(ccpd_path, file_path)
        # print(file_path)
        ccpd = CCPD(file_path)
        img = cv2.imread(file_path)
        mask = ccpd.CCPD_to_Mask(save_path)

        if show_num:
            show_num = show_num - 1
            show_img = np.concatenate((img, mask, np.bitwise_and(img, mask)), axis=1)
            plt.imshow(show_img)
            plt.show()
