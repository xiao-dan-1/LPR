import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils.LoadDataset import _Loadimgs
from tensorflow.keras import models


def UNet_Precdict(model_path, img_path, out_path):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    # 加载图片
    imgs = _Loadimgs(img_path)
    print(len(imgs), imgs.shape)

    # 加载模型
    unet = models.load_model(model_path)
    imgs_mask = unet.predict(imgs)
    return imgs, imgs_mask.astype(np.uint8)


def show(imgs, imgs_mask, out_path):
    i = 0
    for img, img_mask in zip(imgs, imgs_mask):
        i = i + 1
        print(img.shape, img.dtype)
        print(img_mask.shape, img_mask.dtype)
        cv2.imwrite(f"{out_path}/{i}.jpg", np.concatenate((img, img_mask, np.bitwise_and(img, img_mask)), axis=1))
        if i >= 5:
            continue
        plt.imshow(np.concatenate((img, img_mask, np.bitwise_and(img, img_mask)), axis=1))
        plt.show()


if __name__ == "__main__":
    img_path = "datasets/label-studio/images/"
    model_path = "saved_models/unet.h5"
    out_path = "result"
    # 预测分割
    imgs, imgs_mask = UNet_Precdict(model_path, img_path, out_path)
    # 展示
    show(imgs, imgs_mask, out_path)
