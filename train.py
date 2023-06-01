import os.path

import matplotlib.pyplot as plt
import tensorflow as tf
from models.UNet import UNet
from utils.LoadDataset import LoadDataset_From_CCPD
# from utils.Progressbar import Progressbar
from sklearn.model_selection import train_test_split


def show_history(history):
    """history包含以下几个属性：
  训练集loss： loss
  测试集loss： val_loss
  训练集准确率： sparse_categorical_accuracy
  测试集准确率： val_sparse_categorical_accuracy"""
    # print(history.history.keys())
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

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


def train(save_path):
    if not os.path.exists(os.path.split(save_path)[0]):
        os.makedirs(os.path.split(save_path)[0])
    # 1. 加载数据集
    # dataset_path = 'datasets/label-studio/'
    ccpd_path = "D:\Desktop\license plate recognition\CCPD\CCPD2019"

    dataset_path = os.path.join(ccpd_path, r"splits\train.txt")
    data_images, data_labels = LoadDataset_From_CCPD(dataset_path)

    train_images, test_images, train_labels, test_labels = train_test_split(data_images, data_labels, test_size=0.3)
    # 2. 加载模型
    batch_size = 8
    # 断点续训
    checkpoint_path = "checkpoints/unet-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    # 检查生成的检查点并选择最新检查点
    latest = tf.train.latest_checkpoint(checkpoint_dir)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1,
                                                     save_freq=100 * int(len(train_images) / batch_size),
                                                     )
    # 实例化模型
    model = UNet()

    model.save_weights(checkpoint_path.format(epoch=0))

    # Loads the weights
    if latest:
        print("load_weights latest")
        model.load_weights(latest)

    # 3.配置训练方法 model.compile  ，优化器，损失函数，评测指标
    # optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)
    # binary_crossentropy categorical_crossentropy
    # tf.keras.losses
    model.compile(optimizer='adam', loss="mean_squared_error", metrics=['accuracy'])
    # 绘制最终的模型框架
    tf.keras.utils.plot_model(model, to_file='images/graph.png', show_shapes=True)

    # 4. 训练模型
    print("开始训练")
    history = model.fit(train_images, train_labels, epochs=5, batch_size=batch_size,
                        validation_data=(test_images, test_labels),
                        validation_freq=1,
                        callbacks=[cp_callback])  # epochs和batch_size看个人情况调整，batch_size不要过大，否则内存容易溢出

    # 5. 保存模型
    model.save(save_path)
    print('unet.h5保存成功!!!')

    # 6. 显示
    show_history(history)


if __name__ == "__main__":
    save_path = "saved_models/my_unet.h5"
    train(save_path)
