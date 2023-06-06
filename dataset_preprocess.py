import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from utils.Progressbar import Progressbar


def load_data(images_path, annotations_path, image_size):
    # 遍历图像文件夹
    num = 1000
    image_files = sorted(os.listdir(images_path))[0:num]  # os.listdir:获取图像名称列表
    annotation_files = sorted(os.listdir(annotations_path))[0:num]  # 排序确保图像文件和标签文件的对应关系是正确
    data = []
    for img_file, ann_file in Progressbar(zip(image_files, annotation_files), str="加载数据集"):
        img_path = os.path.join(images_path, img_file)
        ann_path = os.path.join(annotations_path, ann_file)

        # 读取图像和标签
        image = cv2.imread(img_path)
        annotation = cv2.imread(ann_path, cv2.IMREAD_GRAYSCALE)  # 读入单通道灰度图

        # 图像预处理
        image = cv2.resize(image, image_size)
        annotation = cv2.resize(annotation, image_size)

        # 将图像和标签添加到数据列表
        data.append((image, annotation))

    return data


def split_dataset(data, test_size=0.2, random_state=42):
    # 拆分数据集为训练集和测试集
    images = np.array([item[0] for item in data])
    annotations = np.array([item[1] for item in data])

    # 标签需要转换为二进制形式，便于使用交叉熵损失函数
    annotations = np.where(annotations > 0, 1, 0)

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(images, annotations, test_size=test_size,
                                                        random_state=random_state)

    return X_train, X_test, y_train, y_test


def apply_mask(image, mask):
    # 将图像和标签掩码应用于分割
    # print(image.dtype, image.shape)
    # print(mask.dtype, mask.shape)
    segmented_image = cv2.bitwise_and(image, image, mask=mask.astype(np.uint8))
    return segmented_image


def visualize_segmentation(image, mask, segmented_image):
    # 可视化分割结果
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(image)
    axes[0].set_title('Original Image')

    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Mask')

    axes[2].imshow(segmented_image)
    axes[2].set_title('Segmented Image')

    plt.show()


def get_weight(labels):
    mean_weight_for_0 = 0
    mean_weight_for_1 = 0
    for label in labels:
        lenght = len(label.flatten())
        label_1 = sum(label.flatten())
        label_0 = lenght - sum(label.flatten())
        weight_for_0 = label_0 / lenght
        weight_for_1 = label_1 / lenght

        mean_weight_for_0 += weight_for_0
        mean_weight_for_1 += weight_for_1

    weight_for_0 = mean_weight_for_0 / len(y_train)
    weight_for_1 = mean_weight_for_1 / len(y_train)
    class_weight = {0: weight_for_0, 1: weight_for_1}
    return class_weight


if __name__ == '__main__':
    data_path = 'datasets/label-studio'
    # data_path = "datasets/CCPD"
    image_folder = 'Images'
    mask_folder = 'Mask'
    image_size = (512, 512)
    validation_split = 0.2

    # 构建数据集路径
    images_path = os.path.join(data_path, image_folder)
    annotations_path = os.path.join(data_path, mask_folder)

    # 加载原始数据
    data = load_data(images_path, annotations_path, image_size)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = split_dataset(data, test_size=validation_split, random_state=42)

    # class_weight = get_weight(y_train)
    # print(class_weight)

    print(X_train.shape, X_train.dtype)
    print(y_train.shape, y_train.dtype)
    # 选择一个样本进行测试
    index = 0
    sample_image = X_train[index]
    sample_mask = y_train[index]

    # 应用标签掩码来分割图像
    segmented_image = apply_mask(sample_image, sample_mask)

    # 可视化分割结果
    visualize_segmentation(sample_image, sample_mask, segmented_image)
