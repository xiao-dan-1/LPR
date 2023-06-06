import datetime
import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from models.CNN import CNN
from utils.LoadDataset import LoadDataset_for_CNN

"""
TF_CPP_MIN_LOG_LEVEL        base_loging	    屏蔽信息	                    输出信息
        “0”	                INFO	        无	                        INFO + WARNING + ERROR + FATAL
        “1”	                WARNING	        INFO	                    WARNING + ERROR + FATAL
        “2”	                ERROR	        INFO + WARNING	            ERROR + FATAL
        “3”	                FATAL	        INFO + WARNING + ERROR	    FATAL
"""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

characters = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫",
              "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "0", "1", "2",
              "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M",
              "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]


def show_history(history):
    """history包含以下几个属性：
  训练集loss： loss
  测试集loss： val_loss
  训练集准确率： sparse_categorical_accuracy
  测试集准确率： val_sparse_categorical_accuracy"""
    print(history.history.keys())

    """dict_keys(['loss', 'dense_loss', 'dense_1_loss', 'dense_2_loss', 'dense_3_loss', 'dense_4_loss', 'dense_5_loss',
               'dense_6_loss', 'dense_accuracy', 'dense_1_accuracy', 'dense_2_accuracy', 'dense_3_accuracy',
               'dense_4_accuracy', 'dense_5_accuracy', 'dense_6_accuracy', 'val_loss', 'val_dense_loss',
               'val_dense_1_loss', 'val_dense_2_loss', 'val_dense_3_loss', 'val_dense_4_loss', 'val_dense_5_loss',
               'val_dense_6_loss', 'val_dense_accuracy', 'val_dense_1_accuracy', 'val_dense_2_accuracy',
               'val_dense_3_accuracy', 'val_dense_4_accuracy', 'val_dense_5_accuracy', 'val_dense_6_accuracy'])
    """
    acc = history.history['dense_accuracy']
    val_acc = history.history['val_dense_accuracy']
    loss = history.history['dense_loss']
    val_loss = history.history['val_dense_loss']
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('raining and validation Accuracy')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('raining and validation Loss')
    plt.legend()
    plt.show()


def train(save_path):
    epochs = 150
    batch_size = 32

    dirname = os.path.dirname(save_path)
    if not os.path.exists(dirname):
        print(f"create {dirname} dir")
        os.makedirs(dirname)
    # 加载数据集
    path = r"D:\Desktop\license plate recognition\CCPD\CCPD2019\lp"
    data_images, data_labels = LoadDataset_for_CNN(path, 10000)
    train_images, test_images, train_labels, test_labels = train_test_split(data_images, data_labels, test_size=0.3)

    train_labels = [train_labels[:, i] for i in range(7)]
    test_labels = [test_labels[:, i] for i in range(7)]
    print(f"data_labels:{data_labels}")
    # 创建实例
    model = CNN()
    # 配置模型
    "model.compile # 配置训练方法 ，优化器，损失函数，评测指标"
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    # 展示模型
    model.summary()
    # tensorboard --logdir logs/fit
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    # 模型训练
    print("开始训练cnn")
    history = model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size,
                        validation_data=(test_images, test_labels),
                        validation_freq=1,
                        callbacks=[tensorboard_callback])  # 总loss为7个loss的和
    model.save(save_path)
    print(f'{save_path}保存成功!!!')

    # 展示
    show_history(history)


def predict_demo(cnn, imgs):
    preds = cnn.predict(imgs)  # 预测形状应为(1,80,240,3)
    lps = [[np.argmax(pre) for pre in pred] for pred in preds]
    # lps = []
    # for pred in preds:
    #     lp = []
    #     for pre in pred:
    #         lp.append(np.argmax(pre))
    #     lps.append(lp)
    return np.array(lps)


if __name__ == '__main__':
    model_save_path = "saved_models/cnn.h5"
    train(model_save_path)
