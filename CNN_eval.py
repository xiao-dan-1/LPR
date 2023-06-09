import tensorflow as tf
# from datasets.CBLPRD.LoadDatasets import *
from datasets.CCPD.LoadDatasets import *

if __name__ == '__main__':
    # 加载验证集
    # datasets_path = "datasets/lp"
    datasets_path = "datasets/CBLPRD"
    datasets_path = "datasets/CCPD/good_lps"
    _, data_images, _, test_labels = LoadData(datasets_path)
    test_labels = [test_labels[:, i] for i in range(7)]
    # 加载模型
    model = tf.keras.models.load_model("saved_models/cnn/my_cnn.h5")
    # 评估模型
    results = model.evaluate(data_images, test_labels)

    print("test loss, test acc:", results)
