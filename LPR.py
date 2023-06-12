import os

import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from CNN_predict import cnn_predict, result_show
from utils.LP_Correction import *
from utils.Progressbar import Progressbar

plt.rcParams['font.sans-serif'] = ['SimHei']
if __name__ == '__main__':
    data_path = "../CCPD/my_use"
    # data_path = "datasets/label-studio"
    image_folder = 'Images'
    model_path = "saved_models/unet/65-gunet.h5"
    image_size = (512, 512)
    image_names = os.listdir(os.path.join(data_path, image_folder))
    image_names = np.random.choice(image_names, 10)
    # 加载待识别图片
    images = []
    for image_name in Progressbar(image_names, str="分割网络加载数据"):
        img_path = os.path.join(data_path, image_folder, image_name)
        # 划分训练集和验证集
        image = cv2.imread(img_path)
        # 图像预处理
        image = np.array(cv2.resize(image, image_size))
        images.append(image)
    images = np.array(images)
    # 加载unet,获得车牌
    model = tf.keras.models.load_model(model_path)
    masks = model.predict(images)
    masks = np.squeeze(masks, axis=-1)  # 降维 去最后一维
    masks = np.round(masks).astype(np.uint8)  # 二值化，数据类型转换
    # 多获取的车牌预处理
    cor_lps, cor_lp_masks = get_lp(images, masks)

    # 显示车牌与车牌掩码
    for image, mask, lp, lp_mask in Progressbar(zip(images, masks, cor_lps, cor_lp_masks)):
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        # for l, lp_m in zip(lp, lp_mask):
        l = lp
        lp_m = lp_mask
        lp_m = cv2.cvtColor(lp_m, cv2.COLOR_GRAY2BGR)
        plt.subplot(2, 1, 1)
        plt.axis("off")  # 取消坐标轴
        plt.title("car")
        plt.imshow(np.concatenate((image, mask * 255), axis=1))
        plt.subplot(2, 1, 2)
        plt.axis("off")  # 取消坐标轴
        plt.title("mask")
        plt.imshow(np.concatenate((l, lp_m), axis=1)[:, :, [2, 1, 0]])
        plt.show()
    # 加载cnn，识别车牌
    # print(cor_lps[0][0].shape)

    # model_path = "saved_models/cnn/best_48_from_goodlps.h5"
    # model = tf.keras.models.load_model(model_path)
    # for lp in cor_lps:
    #     # for l in lp:
    #         # l = cv2.GaussianBlur(l, (3, 3), 0)
    #     l = lp
    #     l = cv2.resize(l, (128, 48), interpolation=cv2.INTER_AREA)  # 128 48
    #     l = np.array([l], dtype=np.uint8)
    #     pres, probabilitys = cnn_predict(model, l)
    #     result_show(l, pres, probabilitys, "result/lpr")
