import numpy as np
import tensorflow as tf
from models.CNN_class import CNN
from utils.LoadDataset import LoadDataset_for_CNN


def train_cnn_demo():
    # 读取数据集
    path = r"D:\Desktop\license plate recognition\CCPD\CCPD2019\lp"
    data_images, data_labels = LoadDataset_for_CNN(path)
    data_labels = [data_labels[:, i] for i in range(7)]
    print(f"data_labels:{data_labels}")
    # 创建实例
    model = CNN()
    # 配置模型
    "model.compile # 配置训练方法 ，优化器，损失函数，评测指标"
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    model.summary()

    # 模型训练
    print("开始训练cnn")
    model.fit(data_images, data_labels, epochs=500, batch_size=32)  # 总loss为7个loss的和
    model.save('my_cnn.h5')
    print('my_cnn.h5保存成功!!!')


def train_cnn_class_demo():
    # 读取数据集
    path = r"D:\Desktop\license plate recognition\CCPD\CCPD2019\lp"
    data_images, data_labels = LoadDataset_for_CNN(path)
    data_labels = [data_labels[:, i] for i in range(7)]
    print(f"data_labels:{data_labels}")

    # 创建实例
    model = CNN()
    # 配置模型
    "model.compile # 配置训练方法 ，优化器，损失函数，评测指标"
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    # 模型训练
    print("开始训练cnn")
    model.fit(data_images.astype(np.float32), data_labels, epochs=100, batch_size=32)  # 总loss为7个loss的和
    model.save("saved_models/cnn_class_model")
    print('cnn_class保存成功!!!')
    model.summary()


if __name__ == '__main__':
    train_cnn_demo()