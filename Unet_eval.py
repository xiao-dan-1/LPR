import os

import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
import tensorflow as tf

from utils.LoadDatasets import LoadData, split_dataset
import logging

# 配置日志记录器
logging.basicConfig(level=logging.INFO,
                    handlers=[logging.FileHandler('logs/evaluation.log'),  # 日志写入文件
                              logging.StreamHandler()],  # 日志打印到终端
                    format='%(asctime)s %(name)s:%(levelname)s - %(message)s')

# 禁用TensorFlow日志
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def evaluate_model(model, X_test, y_test):
    # 预测标签
    y_preds = model.predict(X_test)

    # 将预测标签转换为二进制形式
    y_preds = np.where(y_preds > 0.5, 1, 0)
    y_preds = np.squeeze(y_preds, axis=-1)

    """
    Accuracy（准确率）：衡量模型预测正确的样本比例。
    Precision（精确率）：衡量模型预测为正类的样本中实际为正类的比例。
    Recall（召回率）：衡量实际为正类的样本中模型预测为正类的比例。
    F1 Score（F1得分）：综合衡量精确率和召回率的平衡指标。
    ROC AUC（ROC曲线下的面积）：衡量二分类模型的预测能力。
    Mean IoU（平均交并比）：用于语义分割任务，衡量预测结果与真实标签的重叠程度。
    """
    # 计算准确率、计算精确率、召回率和F1分数
    accuracy = accuracy_score(y_test.flatten(), y_preds.flatten())
    precision = precision_score(y_test.flatten(), y_preds.flatten())
    recall = recall_score(y_test.flatten(), y_preds.flatten())
    f1 = f1_score(y_test.flatten(), y_preds.flatten())

    # 计算Mean IoU
    intersection = np.logical_and(y_test.flatten(), y_preds.flatten())
    union = np.logical_or(y_test.flatten(), y_preds.flatten())
    mean_iou = np.sum(intersection) / np.sum(union)

    # 计算ROC AUC
    # roc_auc = roc_auc_score(y_test.flatten(), y_preds.flatten())

    # 计算ROC曲线
    fpr, tpr, thresholds = roc_curve(y_test.flatten(), y_preds.flatten())
    roc_auc = auc(fpr, tpr)

    # 绘制ROC曲线
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    template = "|   Accuracy:{:.4f}     |   Precision:{:.4f}    |" \
               "    Recall:{:.4f}       |   F1 Score:{:.4f}     |" \
               "    Mean IoU:{:.4f}     |   ROC AUC:{:.4f}      |"
    # 输出评估结果
    evaluation_result = template.format(accuracy, precision, recall, f1, mean_iou, roc_auc)
    logging.info(evaluation_result)
    # 可视化预测结果
    n_samples = 5
    sample_indices = np.random.choice(len(X_test), size=n_samples)

    print(sample_indices)

    for i, sample_index in enumerate(sample_indices):
        # 获取样本图像和真实标签
        sample_image = X_test[sample_index]
        true_label = y_test[sample_index].astype(np.uint8)
        pred_label = y_preds[sample_index].astype(np.uint8)

        # 可视化图像和标签
        plt.subplot(3, n_samples, i + 5 * 0 + 1)
        plt.imshow(sample_image)
        plt.title("image")
        plt.axis("off")  # 取消坐标轴
        plt.subplot(3, n_samples, i + 5 * 1 + 1)
        plt.imshow(cv2.bitwise_and(sample_image, sample_image, mask=true_label))
        plt.title("true")
        plt.axis("off")  # 取消坐标轴
        plt.subplot(3, n_samples, i + 5 * 2 + 1)
        plt.imshow(cv2.bitwise_and(sample_image, sample_image, mask=pred_label))
        plt.title("pred")
        plt.axis("off")  # 取消坐标轴
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # data_path = 'datasets/label-studio'
    data_path = "../CCPD/my_use"
    image_folder = 'Images'
    mask_folder = 'Mask'
    images_path = os.path.join(data_path, image_folder)
    annotations_path = os.path.join(data_path, mask_folder)
    model_path = "saved_models/unet/best.h5"
    image_size = (512, 512)
    validation_split = 0.2

    # 加载数据集
    _, X_test, _, y_test = LoadData(images_path, annotations_path, image_size, num=100, validation_split=0.9)
    print(X_test.shape)
    # 加载模型
    model = tf.keras.models.load_model(model_path, compile=False)
    # 编译模型
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[tf.keras.metrics.BinaryAccuracy()])
    # 评估模型
    evaluate_model(model, X_test, y_test)
