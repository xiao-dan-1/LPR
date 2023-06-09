import os

import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate

from utils.LoadDatasets import LoadData
import albumentations as A


# 定义UNet模型
def unet_model(input_shape, depth=4, filters=16):
    # 输入层
    inputs = Input(input_shape)

    def conv_block(inputs, filters):
        conv1 = Conv2D(filters, 3, activation='relu', padding='same')(inputs)
        conv2 = Conv2D(filters, 3, activation='relu', padding='same')(conv1)
        pool = MaxPooling2D(pool_size=(2, 2))(conv2)
        return conv2, pool

    def upconv_block(inputs, skip_features, filters):
        upsample = Conv2DTranspose(filters, 2, strides=(2, 2), padding='same')(inputs)
        concat = concatenate([skip_features, upsample], axis=3)
        conv1 = Conv2D(filters, 3, activation='relu', padding='same')(concat)
        conv2 = Conv2D(filters, 3, activation='relu', padding='same')(conv1)
        return conv2

    skip_features = []
    x = inputs
    # 编码器部分
    for _ in range(depth):
        conv, x = conv_block(x, filters)
        skip_features.append(conv)
        filters *= 2

    x = Conv2D(filters, 3, activation='relu', padding='same')(x)
    x = Conv2D(filters, 3, activation='relu', padding='same')(x)

    # 解码器部分
    filters //= 2  # //取整数
    for i in range(depth - 1, -1, -1):
        x = upconv_block(x, skip_features[i], filters)
        filters //= 2

    # 输出层
    outputs = Conv2D(1, 1, activation='sigmoid')(x)

    # 创建模型
    model = Model(inputs=inputs, outputs=outputs)

    return model


# 训练模型
def train_model(model, X_train, X_test, y_train, y_test, batch_size, num_epochs,
                checkpoint_path=None):
    # 自定义损失函数（加权交叉熵）
    def weighted_crossentropy_loss(y_true, y_pred):
        # num_plates =   # 车牌样本的数量
        # num_non_plates = np.sum(img == 0)  # 非车牌样本的数量
        pos_weight = np.sum(y_true == 1)  # 正样本的权重
        neg_weight = np.sum(y_true == 0)  # 负样本的权重

        weight_ratio = neg_weight / (neg_weight + pos_weight)
        pos_weight = weight_ratio * 100 * 0.5
        neg_weight = 1
        print(pos_weight, neg_weight)
        y_true = tf.cast(np.expand_dims(y_true, axis=-1), tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        # 计算加权交叉熵损失
        loss = tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, pos_weight, neg_weight)
        return tf.reduce_mean(loss)

    # 定义损失函数和优化器
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    # loss_fn = weighted_crossentropy_loss
    optimizer = tf.keras.optimizers.Adam()

    # 创建指标来跟踪训练和验证的损失和准确性
    train_loss = tf.keras.metrics.Mean()
    train_accuracy = tf.keras.metrics.BinaryAccuracy()
    val_loss = tf.keras.metrics.Mean()
    val_accuracy = tf.keras.metrics.BinaryAccuracy()

    # 计算样本权重
    class_weights = {0: 0.5, 1: 4000}  # 类别权重，根据实际情况设置
    #
    # # 转换类别权重为张量
    class_weights_tensor = tf.constant(list(class_weights.values()), dtype=tf.float32)

    # 创建批量数据生成器
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size).prefetch(1)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size).prefetch(1)

    # 训练循环
    for epoch in range(num_epochs):
        # 重置指标
        train_loss.reset_states()
        train_accuracy.reset_states()
        val_loss.reset_states()
        val_accuracy.reset_states()

        # 训练数据
        for images, labels in train_dataset:
            with tf.GradientTape() as tape:
                # 前向传播
                predictions = model(images, training=True)
                # 计算损失
                # loss_value = loss_fn(labels, predictions)
                loss_value = loss_fn(labels, predictions)
                weighted_loss = tf.reduce_mean(loss_value * tf.gather(class_weights_tensor, tf.cast(labels, tf.int32)))
                weighted_loss = loss_value
            # 计算梯度
            grads = tape.gradient(weighted_loss, model.trainable_variables)
            # 更新模型参数
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            # 更新训练指标
            train_loss(weighted_loss)
            train_accuracy(labels, predictions)

        # 验证数据
        for val_images, val_labels in val_dataset:
            val_predictions = model(val_images, training=False)
            val_loss_value = loss_fn(val_labels, val_predictions)
            val_loss(val_loss_value)
            val_accuracy(val_labels, val_predictions)

        # 打印训练和验证指标
        template = 'Epoch {}, Loss: {}, Accuracy: {}, Val Loss: {}, Val Accuracy: {}'
        print(template.format(epoch + 1,
                              train_loss.result(),
                              train_accuracy.result() * 100,
                              val_loss.result(),
                              val_accuracy.result() * 100))

        if checkpoint_path and (epoch + 1) % 5 == 0:
            root, name = os.path.split(checkpoint_path)
            checkpoint_path = os.path.join(root, f"{epoch}-{name}")
            # 保存模型
            print(f"saved {checkpoint_path} model")
            model.save(checkpoint_path)
    if checkpoint_path:
        print(f"saved {checkpoint_path} last model")
        model.save(checkpoint_path)
    # 返回训练后的模型
    return model


# 评估模型（用于超参数调优）
def evaluate_model(model, X_test, y_test):
    def calculate_mean_iou(y_true, y_pred):
        intersection = np.logical_and(y_true, y_pred)
        union = np.logical_or(y_true, y_pred)
        iou_score = np.sum(intersection) / np.sum(union)
        return iou_score

    # 预测标签
    y_preds = model.predict(X_test)

    # 将预测标签转换为二进制形式
    y_preds = np.where(y_preds > 0.5, 1, 0)
    y_preds = np.squeeze(y_preds, axis=-1)

    # 计算准确率、精确率、召回率、F1分数、Mean IoU和ROC AUC
    accuracy = accuracy_score(y_test.flatten(), y_preds.flatten())
    precision = precision_score(y_test.flatten(), y_preds.flatten())
    recall = recall_score(y_test.flatten(), y_preds.flatten())
    f1 = f1_score(y_test.flatten(), y_preds.flatten())
    iou = calculate_mean_iou(y_test, y_preds)  # 自定义函数，计算Mean IoU
    roc_auc = roc_auc_score(y_test.flatten(), y_preds.flatten())

    # 返回评估指标字典
    evaluation_results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou,
        'roc_auc': roc_auc
    }

    return evaluation_results


# 图像增强
def data_augment(X_train, y_train):
    transform = A.Compose([
        A.RandomCrop(height=512, width=512),  # 随机裁剪
        A.HorizontalFlip(p=0.5),  # 随机水平翻转
        A.Rotate(limit=15, p=0.5),  # 限制较小的旋转角度
        A.RandomScale(scale_limit=0.1),  # 限制较小的缩放范围
        A.ShiftScaleRotate(scale_limit=(1.0, 1.5), interpolation=0, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1),  # 限制较小的亮度和对比度调整范围
        A.GaussianBlur(blur_limit=(3, 7)),  # 增加模糊程度，可自定义模糊核大小
        A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10),  # 限制较小的颜色偏移
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10),  # 限制较小的色相、饱和度和明度调整范围
        A.Resize(image_size[0], image_size[1]),
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
def show_augmented_result_demo(X_train, X_train_augmented, y_train_augmented):
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

    # 选择样本进行测试
    index = 5
    sample_images_augs = X_train_augmented[0:index]
    sample_images = X_train[0:index]
    sample_masks = y_train_augmented[0:index]

    # 应用标签掩码来分割图像
    for sample_image, sample_images_aug, sample_mask in zip(sample_images, sample_images_augs, sample_masks):
        segmented_image = cv2.bitwise_and(sample_images_aug, sample_images_aug, mask=sample_mask.astype(np.uint8))
        # 可视化分割结果
        visualize_segmentation(sample_images_aug, sample_mask, segmented_image)
        visualize_segmentation(sample_image, sample_mask, segmented_image)


# def objective(trial):
#     # 定义超参数搜索空间
#     learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
#     depth = trial.suggest_int('depth', 1, 5)  # 调整上采样和下采样的次数
#     filters = trial.suggest_int('filters', [2, 4, 8, 16, 32])
#     batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
#     num_epochs = trial.suggest_int('num_epochs', 10, 100, step=10)
#
#     # 创建模型
#     model = unet_model(input_shape=(512, 512, 3), depth=depth, filters=filters)
#
#     # 训练模型
#     trained_model = train_model(model, X_train, X_test, y_train, y_test, batch_size, num_epochs, learning_rate)
#
#     # 评估模型
#     evaluation_result = evaluate_model(trained_model, X_test, y_test)
#
#     # 返回评估指标（平均交并比）作为目标函数值
#     return evaluation_result['iou']


if __name__ == '__main__':
    # 基本参数
    data_path = "../CCPD/my_use"
    image_folder = 'Images'
    mask_folder = 'Mask'
    images_path = os.path.join(data_path, image_folder)
    annotations_path = os.path.join(data_path, mask_folder)
    save_model_path = "saved_models/unet/gunet.h5"
    input_shape = (512, 512, 3)
    image_size = (512, 512)
    validation_split = 0.2

    learning_rate = 0.001
    depth = 4
    filters = 16
    batch_size = 4
    num_epochs = 500

    # 加载数据集
    X_train, X_test, y_train, y_test = LoadData(images_path, annotations_path, image_size,
                                                validation_split=validation_split, num=2000)
    print(X_train.shape, X_train.dtype)
    print(y_train.shape, y_train.dtype)
    print(X_test.shape, X_test.dtype)
    print(y_test.shape, y_test.dtype)

    # 图像增强
    X_train_augmented, y_train_augmented = data_augment(X_train, y_train)
    show_augmented_result_demo(X_train, X_train_augmented, y_train_augmented)  # show_augmented_result_demo

    # 创建模型
    model = unet_model(input_shape=input_shape, depth=depth, filters=filters)

    # # 训练
    train_model(model, X_train_augmented, X_test, y_train_augmented, y_test, batch_size, num_epochs,
                checkpoint_path=save_model_path)
