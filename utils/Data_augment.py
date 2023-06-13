import os, sys

dir_mytest = "D:\Desktop\license plate recognition\My_LPR"
sys.path.insert(0, dir_mytest)

import albumentations as A
import numpy as np

from matplotlib import pyplot as plt
from utils.LoadDatasets import LoadData


def data_augment(X_train, y_train, image_size):
    transform = A.Compose([
        A.CenterCrop(image_size[0] // 2, image_size[1] // 2, p=0.9),  # 中心裁剪
        A.Resize(image_size[0], image_size[1]),  # 尺寸调节
        # A.RandomSizedCrop(min_max_height=(100, 400), height=image_size[0] // 3, width=image_size[1] // 3, p=1),  # 随机大小裁剪
        # A.RandomCrop(height=512, width=512),    # 随机裁剪

        A.OneOf([
            A.Rotate(limit=20, p=0.8),  # 限制较小的旋转角度
            A.HorizontalFlip(p=0.8),  # 随机水平翻转
            A.VerticalFlip(p=0.8),  # 垂直旋转
            # A.Transpose(p=0.8),  # 转置
        ], p=1),
        A.RandomScale(scale_limit=0.1),  # 限制较小的缩放范围
        A.PadIfNeeded(min_height=image_size[0], min_width=image_size[1], p=1),  # 填充 默认反射填充。

        A.OneOf([
            A.ElasticTransform(alpha=40, sigma=120 * 0.05, alpha_affine=40 * 0.03, p=0.8),  # 弹性变化
            A.GridDistortion(p=0.8),  # 网格失真
            # A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=0.8),  # 光学失真
        ], p=1),

        A.OneOf([
            A.MedianBlur(blur_limit=5, always_apply=False, p=0.8),  # 中值滤波
            A.Blur(blur_limit=5, always_apply=False, p=0.8),  # 均值滤波
            A.GaussianBlur(blur_limit=5, always_apply=False, p=0.8),  # 高斯滤波
        ], p=1),

        A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10),  # 限制较小的颜色偏移
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1),  # 限制较小的亮度和对比度调整范围
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10),  # 限制较小的色相、饱和度和明度调整范围

        A.Resize(image_size[0], image_size[1]),  # 尺寸调节
        # A.ShiftScaleRotate(scale_limit=(1.0, 1.5), interpolation=0, p=0.5),
        # 添加其他所需的增强变换
    ])

    X_train_augmented = []
    y_train_augmented = []

    for x, y in zip(X_train, y_train):
        augmented = transform(image=x, mask=y)
        image = augmented["image"]
        mask = augmented["mask"]
        # print("image:", image.shape, image.dtype)
        # print("mask:", mask.shape, mask.dtype)
        X_train_augmented.append(image)
        y_train_augmented.append(mask)

    X_train_augmented = np.array(X_train_augmented)
    y_train_augmented = np.array(y_train_augmented)
    return X_train_augmented, y_train_augmented


# 图像增强效果显示
def show_augmented_result_demo(X_train, y_train, X_train_augmented, y_train_augmented):
    def visualize_segmentation(image, mask, sample_images_aug, sample_masks_aug):
        # # 可视化分割结果
        # fig, axes = plt.subplots(2, 2, figsize=(12, 4))
        plt.subplot(2, 2, 1)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis("off")  # 取消坐标轴
        plt.subplot(2, 2, 2)
        plt.imshow(mask, cmap='gray')
        plt.title('Mask')
        plt.axis("off")  # 取消坐标轴
        plt.subplot(2, 2, 3)
        plt.imshow(sample_images_aug)
        plt.title('sample_images_aug')
        plt.axis("off")  # 取消坐标轴
        plt.subplot(2, 2, 4)
        plt.imshow(sample_masks_aug)
        plt.title('sample_masks_aug')
        plt.axis("off")  # 取消坐标轴
        plt.tight_layout()
        plt.show()

    # 选择样本进行测试
    index = 5
    sample_images = X_train[0:index]
    sample_masks = y_train[0:index]
    sample_images_augs = X_train_augmented[0:index]
    sample_masks_augs = y_train_augmented[0:index]

    # 应用标签掩码来分割图像
    for sample_image, sample_mask, sample_images_aug, sample_masks_aug in zip(sample_images, sample_masks,
                                                                              sample_images_augs, sample_masks_augs):
        # 可视化分割结果
        visualize_segmentation(sample_image, sample_mask, sample_images_aug, sample_masks_aug)


# demo
if __name__ == '__main__':
    data_path = "../../CCPD/my_use"
    image_folder = 'Images'
    mask_folder = 'Mask'
    images_path = os.path.join(data_path, image_folder)
    annotations_path = os.path.join(data_path, mask_folder)
    image_size = (512, 512)
    validation_split = 0.2
    # 加载数据集
    X_train, X_test, y_train, y_test = LoadData(images_path, annotations_path, image_size,
                                                validation_split=validation_split, num=10)

    X_train_augmented, y_train_augmented = data_augment(X_train, y_train, image_size)
    show_augmented_result_demo(X_train, y_train, X_train_augmented, y_train_augmented)  # show_augmented_result_demo
