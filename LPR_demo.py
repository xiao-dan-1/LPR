import os

import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from CNN_predict import cnn_predict, result_show
from utils.LP_Correction import *
from utils.Progressbar import Progressbar

plt.rcParams['font.sans-serif'] = ['SimHei']

# 模型路径
unet_model_path = "saved_models/unet/65-gunet.h5"
cnn_model_path = "saved_models/cnn/best_48_from_goodlps.h5"
image_size = (512, 512)
# 加载UNet模型
unet_model = tf.keras.models.load_model(unet_model_path)
# 加载CNN模型
cnn_model = tf.keras.models.load_model(cnn_model_path)


def get_mask(image):
    # 预测
    masks = unet_model.predict(np.array([image]))
    masks = np.squeeze(masks)  # 降维
    mask = np.where(masks > 0.5, 255, 0).astype(np.uint8)  # 二值化，数据类型转换
    return mask


def get_lp_fuc(image, mask):
    cor_lps, cor_lp_masks = get_lp(np.array([image]), np.array([mask]))
    cor_lp = np.squeeze(cor_lps)  # 降维
    return cor_lp


def predict(cor_lp):
    cor_lp = cv2.resize(cor_lp, (128, 48), interpolation=cv2.INTER_AREA)  # 128 48
    cor_lp = np.array([cor_lp], dtype=np.uint8)

    pre, probability = cnn_predict(cnn_model, cor_lp)
    acc_num = np.sum(probability > 0.9)
    if acc_num < 5:
        pre = "pre:未识别车牌"
        acc = "acc:0"
    else:
        string = f"pre:{pre[0]}acc:{list(probability)[0]}"
        pre = f"pre:{pre[0]}"
        acc = f"acc:{list(probability)[0]}"
    return pre, acc


if __name__ == '__main__':
    while True:
        # 加载待识别图片
        input_path = input("输入待识别车牌路径：")
        image = cv2.imdecode(np.fromfile(input_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        image = cv2.resize(image, image_size).astype(np.uint8)
        image = np.array(image)
        plt.imshow(image)
        plt.title("原始图片")
        plt.show()
        # 加载unet,获得车牌
        print("加载unet,获取掩码中...")
        mask = get_mask(image)
        print("Done")
        plt.title("掩码")
        plt.imshow(mask)
        plt.show()
        # 多获取的车牌预处理
        lp = get_lp_fuc(image, mask)
        plt.title("提取车牌")
        plt.imshow(lp)
        plt.show()
        # 显示车牌与车牌掩码
        print("加载cnn,识别车牌中...")
        pre, acc = predict(lp)
        print("Done")
        print("预测结果：")
        print(pre)
        print(acc)
        print("")
