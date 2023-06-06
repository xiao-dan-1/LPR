import cv2
import numpy as np
from utils.Progressbar import Progressbar
from datasets.CCPD2Datasets import *


def get_files_path(path):
    files_path = []
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            file_path = os.path.join(root, filename)
            files_path.append(file_path)
    return files_path


def _Loadimgs(filepath, height=512, width=512):
    imgs = []
    files_path = get_files_path(filepath) if os.path.isdir(filepath) else [filepath]
    for file_path in files_path:
        img = cv2.imread(file_path)
        if img.shape != (height, width, 3):
            img = cv2.resize(img, dsize=(height, width), interpolation=cv2.INTER_AREA)[:, :, :3]
        # print(img.shape)
        imgs.append(img)
    return np.array(imgs)


def LoadDataset(imgs_path, labels_path):
    img = _Loadimgs(imgs_path)
    label = _Loadimgs(labels_path)
    # img = img.astype('float16')
    # label = label.astype('float32')
    return img, label


def LoadDataset_From_COCO(filepath):
    imgs_path = os.path.join(filepath, 'Images')
    labels_path = os.path.join(filepath, 'Mask')
    imgs, labels = LoadDataset(imgs_path, labels_path)
    return imgs, labels


def LoadDataset_From_CCPD(filepath):
    ccpd_path = os.path.dirname(filepath).replace('\splits', '')
    # print(f"ccpd_path:{ccpd_path}")
    files_path = []
    with open(filepath, 'r') as f:
        for line in f.readlines():
            files_path.append(line.replace('\n', ''))
    imgs = []
    labels = []
    for file_path in Progressbar(files_path[0:100]):
        file_name = os.path.split(file_path)[-1]
        imgs_path = os.path.join(ccpd_path, file_path)
        labels_path = os.path.join(ccpd_path, 'Mask', file_name)
        # print(f"imgs_path:{imgs_path}")
        # print(f"imgs_path:{labels_path}")
        [img], [label] = LoadDataset(imgs_path, labels_path)
        imgs.append(img)
        labels.append(label)
    return np.array(imgs), np.array(labels)


def LoadDataset_for_CNN(filepath=r"D:\Desktop\license plate recognition\CCPD\CCPD2019\lp", num=None, ):
    char_dict = {"京": 0, "沪": 1, "津": 2, "渝": 3, "冀": 4, "晋": 5, "蒙": 6, "辽": 7, "吉": 8, "黑": 9, "苏": 10,
                 "浙": 11, "皖": 12, "闽": 13, "赣": 14, "鲁": 15, "豫": 16, "鄂": 17, "湘": 18, "粤": 19, "桂": 20,
                 "琼": 21, "川": 22, "贵": 23, "云": 24, "藏": 25, "陕": 26, "甘": 27, "青": 28, "宁": 29, "新": 30,
                 "0": 31, "1": 32, "2": 33, "3": 34, "4": 35, "5": 36, "6": 37, "7": 38, "8": 39, "9": 40,
                 "A": 41, "B": 42, "C": 43, "D": 44, "E": 45, "F": 46, "G": 47, "H": 48, "J": 49, "K": 50,
                 "L": 51, "M": 52, "N": 53, "P": 54, "Q": 55, "R": 56, "S": 57, "T": 58, "U": 59, "V": 60,
                 "W": 61, "X": 62, "Y": 63, "Z": 64}

    files_path = get_files_path(filepath) if os.path.isdir(filepath) else [filepath]
    print(f"data len:{len(files_path)}")
    imgs = []
    labels = []
    for file_path in files_path[0:num]:
        file_name = os.path.split(file_path)[-1]
        file_name = os.path.splitext(file_name)[0]
        label = [char_dict[i] for i in file_name]
        img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img.shape != (220, 70, 3):
            img = cv2.resize(img, dsize=(220, 70), interpolation=cv2.INTER_AREA)[:, :, :3]
        # print(file_name)
        # print(img.shape,type(img))
        imgs.append(img)
        labels.append(label)
    return np.array(imgs), np.array(labels)


if __name__ == '__main__':
    imgs, labels = LoadDataset_for_CNN()

    print(f"img:{imgs.shape},{imgs.dtype}")
    print(f"labels:{labels.shape},{labels.dtype}")
