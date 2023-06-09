import os
import sys

dir_mytest = "D:\Desktop\license plate recognition\My_LPR"
sys.path.insert(0, dir_mytest)
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from LoadDatasets import *
from datasets.CCPD.CCPD import CCPD
from utils.LP_Correction import *
from test import *
from utils.Progressbar import Progressbar

plt.rcParams['font.sans-serif'] = ['SimHei']
# 禁用TensorFlow日志
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'

data_path = "../../../CCPD/my_use"
image_folder = 'Images'
model_path = "../../saved_models/unet/gunet.h5"
save_path = "new_lps"
if not os.path.exists(save_path):
    print(f"create {save_path} dir ")
    os.makedirs(save_path)
image_size = (512, 512)
i = 2
for i in range(2, 100 + 1):
    print(f'{i}'.center(60,'-'))
    image_names = os.listdir(os.path.join(data_path, image_folder))[600 * i:600 * (i + 1)]

    # 加载模型
    model = tf.keras.models.load_model(model_path)
    # 加载测试集

    images = []
    labels = []
    for image_name in Progressbar(image_names):
        img_path = os.path.join(data_path, image_folder, image_name)
        ccpd = CCPD(img_path)
        # 划分训练集和验证集
        image = cv2.imread(ccpd.path)
        # 图像预处理
        image = cv2.resize(image, image_size)
        images.append(image)
        labels.append(ccpd.getLP())

    images = np.array(images)
    print(images.shape, images.dtype)
    # 创建批量数据生成器
    test_dataset = tf.data.Dataset.from_tensor_slices((images, labels)).batch(100).prefetch(1)
    for images, labels in Progressbar(test_dataset):
        print(images.shape, images.dtype)
        # 预测
        masks = model.predict(images)
        masks = np.squeeze(masks, axis=-1)  # 降维 去最后一维
        masks = np.round(masks).astype(np.uint8)  # 二值化，数据类型转换
        # 根据掩码获取车牌
        lps, lp_masks = get_lp(images.numpy(), masks)
        # print(lps)
        for lp, label in zip(lps, labels.numpy()):
            label = str(label, 'utf-8')
            # print(label,type(label))
            for l in lp:
                # plt.axis("off")  # 取消坐标轴
                # plt.title(label)
                # plt_show(l[:, :, [2, 1, 0]])
                cv2.imencode('.jpg', l)[1].tofile(f"{save_path}/{label}.jpg")
