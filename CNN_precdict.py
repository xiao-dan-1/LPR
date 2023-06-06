import os

import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from utils.LoadDataset import LoadDataset_for_CNN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

characters = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫",
              "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "0", "1", "2",
              "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M",
              "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]


def cnn_predict(cnn, imgs):
    preds = cnn.predict(imgs)  # 预测形状应为(1,80,240,3)
    lps = [[np.argmax(pre) for pre in pred] for pred in preds]
    # lps = []
    # for pred in preds:
    #     lp = []
    #     for pre in pred:
    #         lp.append(np.argmax(pre))
    #     lps.append(lp)
    return imgs, np.array(lps).T.tolist()


def labels2lps(labels):
    lps = []
    for label in labels:
        lp = ""
        for i in label:
            lp += characters[i]
        lps.append(lp)
    return lps


def result_show(imgs, pres, save_path=None):
    i = 0
    for img, pre in zip(imgs, labels2lps(pres)):
        i = i + 1
        if save_path:
            if not os.path.exists(save_path):
                print(f"create {save_path} dir ")
                os.makedirs(save_path)
            cv2.imencode('.jpg', img)[1].tofile(f"{save_path}/{pre}.jpg")
        if i >= 5:
            continue
        plt.subplot(2, 1, 1)
        plt.imshow(img[:, :, [2, 1, 0]])
        print(pre)
        plt.show()


if __name__ == '__main__':
    # 加载图片
    img_path = "D:/Desktop/license plate recognition/CCPD/CCPD2019/lp/"
    out_path = "result/cnn"
    data_images, data_labels = LoadDataset_for_CNN(img_path, 500)
    # 加载模型
    model_path = "saved_models/cnn.h5"
    model = tf.keras.models.load_model(model_path)
    imgs, pres = cnn_predict(model, data_images)

    # 展示
    result_show(imgs, pres, out_path)
