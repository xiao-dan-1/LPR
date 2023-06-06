import datetime
import os

import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.losses import BinaryFocalCrossentropy
import tensorflow as tf
from dataset_preprocess import *
from dataset_preprocess import load_data, split_dataset
from utils.Progressbar import Progressbar


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

    weight_for_0 = mean_weight_for_0 / len(labels)
    weight_for_1 = mean_weight_for_1 / len(labels)
    class_weight = {0: weight_for_0, 1: weight_for_1}
    return class_weight


class UNet:
    def __init__(self, input_shape, depth, filters):
        self.input_shape = input_shape
        self.depth = depth
        self.filters = filters
        self.model = self.build_model()

    def build_model(self):
        # 输入层
        inputs = Input(self.input_shape)

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

        skip_features = []  # 0 1 2
        x = inputs
        # 编码器部分
        for _ in range(self.depth):
            conv, x = conv_block(x, self.filters)
            skip_features.append(conv)
            self.filters *= 2

        x = Conv2D(self.filters, 3, activation='relu', padding='same')(x)
        x = Conv2D(self.filters, 3, activation='relu', padding='same')(x)

        # 解码器部分
        self.filters //= 2  # //取整数
        for i in range(self.depth - 1, -1, -1):
            x = upconv_block(x, skip_features[i], self.filters)
            self.filters //= 2

        # 输出层
        outputs = Conv2D(1, 1, activation='sigmoid')(x)

        # 创建模型
        model = Model(inputs=inputs, outputs=outputs)

        return model

    def describe_model(self):
        self.model.summary()
        for layer in self.model.layers:
            print(layer.name, layer.input_shape, layer.output_shape)

    def preprocess_data(self, images_path, annotations_path, image_size):
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

    def split_dataset(self, data, test_size=0.2, random_state=42):
        # 拆分数据集为训练集和测试集
        images = np.array([item[0] for item in data])
        annotations = np.array([item[1] for item in data])

        # 标签需要转换为二进制形式，便于使用交叉熵损失函数
        annotations = np.where(annotations > 0, 1, 0)

        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(images, annotations, test_size=test_size,
                                                            random_state=random_state)

        return X_train, X_test, y_train, y_test

    def load_data(self, data_path, image_folder, mask_folder, image_size, test_size=0.2, random_state=None):
        # 构建数据集路径
        images_path = os.path.join(data_path, image_folder)
        annotations_path = os.path.join(data_path, mask_folder)

        # 加载原始数据
        data = self.preprocess_data(images_path, annotations_path, image_size)

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = self.split_dataset(data, test_size, random_state)

        return X_train, X_test, y_train, y_test

    def train(self, X_train, X_test, y_train, y_test, batch_size, num_epochs, save_model_path=None):
        # 定义类别权重
        # class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train),
        #                                                   y=np.ravel(y_train, order='C'))
        # weight = {i: class_weights[i] for i in range(len(class_weights))}

        # 计算样本权重
        # 计算样本权重
        def compute_sample_weight(y, weight_factor):
            sample_weight = np.ones(y.shape, dtype=np.float32)
            sample_weight[y == 1] = weight_factor
            return sample_weight

        # 定义权重因子
        weight_factor = 10  # 示例，可以根据实际情况设置权重因子

        # 计算样本权重
        sample_weights = compute_sample_weight(y_train, weight_factor)
        sample_weights = np.expand_dims(sample_weights, axis=-1)

        def data_generator(X, y, sample_weights, batch_size):
            num_samples = X.shape[0]
            num_batches = num_samples // batch_size

            while True:
                for batch_index in range(num_batches):
                    start = batch_index * batch_size
                    end = (batch_index + 1) * batch_size

                    batch_X = X[start:end]
                    batch_y = y[start:end]
                    batch_sample_weights = sample_weights[start:end]

                    yield (batch_X, batch_y, batch_sample_weights)

        # 假设您已经准备好了 X_train, y_train 和 sample_weights
        # 定义批次大小和训练迭代次数

        # 创建生成器实例
        generator = data_generator(X_train, y_train, sample_weights, batch_size)

        # 编译模型

        # 使用生成器进行训练

        # 断点续训
        # checkpoint_path = "checkpoints/UNet/unet-{epoch:04d}.ckpt"
        # checkpoint_dir = os.path.dirname(checkpoint_path)
        # # 检查生成的检查点并选择最新检查点
        # latest = tf.train.latest_checkpoint(checkpoint_dir)

        # 如果存在最新检查点，则加载模型权重
        # if latest:
        #     self.model.load_weights(latest)

        # 定义回调函数
        # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
        #                                                  save_weights_only=True,
        #                                                  save_best_only=True,
        #                                                  verbose=1,
        #                                                  save_freq=100 * int(len(X_train) / batch_size)
        #                                                  )

        # 编译模型
        model = self.model

        # 配置模型 'binary_crossentropy'
        # focal_loss = BinaryFocalCrossentropy(gamma=2.0)
        # 定义并使用自定义损失函数

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'binary_accuracy'],
                      sample_weight_mode="temporal")

        # tensorboard --logdir logs//unet/
        log_dir = "logs/unet/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        self.history = model.fit(generator, steps_per_epoch=len(X_train) // batch_size, epochs=num_epochs,
                                 validation_data=(X_test, y_test))  # callbacks=[cp_callback, tensorboard_callback])
        self.model = model

        # # 保存模型
        # if save_model_path:
        #     model.save(save_model_path)
        # # 预测
        # predictions = model.predict(X_test)
        #
        # # 选择样本进行测试
        # index = 5
        # sample_images = X_test[0:index]
        # sample_preds = predictions[0:index]
        #
        # # 应用标签掩码来分割图像
        # for sample_image, sample_pred in zip(sample_images, sample_preds):
        #     sample_pred = np.squeeze(sample_pred, axis=-1)  #
        #     sample_pred = np.round(sample_pred).astype(np.uint8)
        #     #
        #     segmented_image = apply_mask(sample_image, sample_pred)
        #     # 可视化分割结果
        #     visualize_segmentation(sample_image, sample_pred, segmented_image)

    def save_model(self, filepath):
        if self.model is None:
            raise ValueError("Model has not been built yet. Please build the model first.")
        self.model.save(filepath)
        print("Model saved successfully.")

    def show_history(self):
        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        plt.subplot(1, 2, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.title('raining and validation Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.title('raining and validation Loss')
        plt.legend()
        plt.show()

# # 创建UNet对象
# unet = UNet(input_shape=(512, 512, 3), depth=2, filters=16)
#
# # 描述模型
# unet.describe_model()
#
# # 定义数据集路径和一些训练参数
# data_path = 'datasets/label-studio'
# image_folder = 'Images'
# mask_folder = 'Mask'
# image_size = (512, 512)
# batch_size = 6
# num_epochs = 10
# validation_split = 0.2
# save_model_path = "saved_models/gunetclass.h5"
#
# # 加载图像和标签数据
# X_train, X_test, y_train, y_test = unet.load_data(data_path, image_folder, mask_folder, image_size,
#                                                   test_size=validation_split, random_state=42)
# # 调用训练方法
# unet.train(X_train, y_train, X_test, y_test, batch_size=batch_size, num_epochs=num_epochs,
#            save_model_path=save_model_path)
#
# #  显示
# unet.show_history()
