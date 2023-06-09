import os, sys

dir_mytest = "D:\Desktop\license plate recognition\My_LPR"
sys.path.insert(0, dir_mytest)
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from utils.Progressbar import Progressbar

plt.rcParams['font.sans-serif'] = ['SimHei']
char_dict = {"京": 0, "沪": 1, "津": 2, "渝": 3, "冀": 4, "晋": 5, "蒙": 6, "辽": 7, "吉": 8, "黑": 9, "苏": 10,
             "浙": 11, "皖": 12, "闽": 13, "赣": 14, "鲁": 15, "豫": 16, "鄂": 17, "湘": 18, "粤": 19, "桂": 20,
             "琼": 21, "川": 22, "贵": 23, "云": 24, "藏": 25, "陕": 26, "甘": 27, "青": 28, "宁": 29, "新": 30,
             "0": 31, "1": 32, "2": 33, "3": 34, "4": 35, "5": 36, "6": 37, "7": 38, "8": 39, "9": 40,
             "A": 41, "B": 42, "C": 43, "D": 44, "E": 45, "F": 46, "G": 47, "H": 48, "J": 49, "K": 50,
             "L": 51, "M": 52, "N": 53, "P": 54, "Q": 55, "R": 56, "S": 57, "T": 58, "U": 59, "V": 60,
             "W": 61, "X": 62, "Y": 63, "Z": 64}

characters = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫",
              "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "0", "1", "2",
              "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M",
              "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]


def lpencode(value):
    label = [char_dict[i] for i in value]
    return label


def lpdecode(value):
    lp = [characters[i] for i in value]
    return lp


def LoadData(data_path, num=None):
    file_names = os.listdir(data_path)
    images = []
    labels = []
    for file_name in Progressbar(file_names[0:num]):
        # print(file_name)
        img_path = os.path.join(data_path, file_name)
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, dsize=(128, 48), interpolation=cv2.INTER_AREA)[:, :, :3]
        images.append(img)
        labels.append(lpencode(os.path.splitext(file_name)[0]))
    images = np.array(images)
    labels = np.array(labels)
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.3)
    return train_images, test_images, train_labels, test_labels


data_path = "D:\Desktop\license plate recognition\My_LPR\datasets/CCPD/new_lps"
if __name__ == '__main__':
    train_images, test_images, train_labels, test_labels = LoadData(data_path)
    print("train_images:", train_images.shape, train_images.dtype)
    print("train_labels:", train_labels.shape, train_labels.dtype)
    print("test_images:", test_images.shape, test_images.dtype)
    print("test_labels:", test_labels.shape, test_labels.dtype)
