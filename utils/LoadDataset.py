import os

import cv2
import numpy as np
from utils.Progressbar import Progressbar


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
    imgs_path = os.path.join(filepath, 'images')
    labels_path = os.path.join(filepath, 'labels')
    LoadDataset(imgs_path, labels_path)


def LoadDataset_From_CCPD(filepath):
    ccpd_path = os.path.dirname(filepath).replace('\splits', '')
    # print(f"ccpd_path:{ccpd_path}")
    files_path = []
    with open(filepath, 'r') as f:
        for line in f.readlines():
            files_path.append(line.replace('\n', ''))
    imgs = []
    labels = []
    for file_path in Progressbar(files_path[0:1000]):
        file_name = os.path.split(file_path)[-1]
        imgs_path = os.path.join(ccpd_path, file_path)
        labels_path = os.path.join(ccpd_path, 'mask', file_name)
        # print(f"imgs_path:{imgs_path}")
        # print(f"imgs_path:{labels_path}")
        [img], [label] = LoadDataset(imgs_path, labels_path)
        imgs.append(img)
        labels.append(label)
    return np.array(imgs), np.array(labels)
