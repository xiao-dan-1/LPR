import os

import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, \
    precision_score, recall_score, f1_score
import numpy as np
from dataset_preprocess import *
from dataset_preprocess import load_data, split_dataset


def evaluate_model_with_visualization(model, X_test, y_test):
    # 进行预测
    y_preds = model.predict(X_test)
    y_preds = np.squeeze(y_preds, axis=-1) #
    y_preds = np.round(y_preds).astype(np.uint8)

    for sample_image, sample_mask in zip(X_test, y_preds):
        # 应用标签掩码来分割图像
        segmented_image = apply_mask(sample_image, sample_mask)
        # 可视化分割结果
        visualize_segmentation(sample_image, sample_mask, segmented_image)


if __name__ == '__main__':
    data_path = 'datasets/label-studio'
    image_folder = 'Images'
    mask_folder = 'Mask'
    model_path = "saved_models/unet/gunet.h5"
    image_size = (512, 512)
    validation_split = 0.2
    # 数据预处理
    images_path = os.path.join(data_path, image_folder)
    annotations_path = os.path.join(data_path, mask_folder)
    data = load_data(images_path, annotations_path, image_size)

    # 划分训练集和验证集
    X_train, X_test, y_train, y_test = split_dataset(data, test_size=validation_split, random_state=42)

    # 加载模型
    model = tf.keras.models.load_model(model_path)

    # 使用加载的模型进行预测或评估
    evaluate_model_with_visualization(model, X_test[0:10], y_test[0:10])
