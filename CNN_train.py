import datetime
import tensorflow as tf
from models.CNN import CNN
# from datasets.lp.LoadDatasets import *
# from datasets.CBLPRD.LoadDatasets import *
from datasets.CCPD.LoadDatasets import *
import albumentations as A

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
    epochs = 100
    batch_size = 32
    num = None
    image_size = (48, 128)

    dirname = os.path.dirname(save_path)
    if not os.path.exists(dirname):
        print(f"create {dirname} dir")
        os.makedirs(dirname)
    # 加载数据集
    # path = r"D:\Desktop\license plate recognition\CCPD\CCPD2019\lp"
    # data_images, data_labels = LoadDataset_for_CNN(path, num)
    # train_images, test_images, train_labels, test_labels = train_test_split(data_images, data_labels, test_size=0.3)

    # datasets_path = "datasets/lp"
    # # datasets_path = "datasets/CBLPRD"
    # train_images, test_images, train_labels, test_labels = LoadData(datasets_path, num=40000)
    # datasets_path = "../CCPD/new_lps"
    datasets_path = "datasets/CCPD/good_lps"
    train_images, test_images, train_labels, test_labels = LoadData(datasets_path,num=num)
    print("train_images:", train_images.shape, train_images.dtype)
    print("train_labels:", train_labels.shape, train_labels.dtype)
    print("test_images:", test_images.shape, test_images.dtype)
    print("test_labels:", test_labels.shape, test_labels.dtype)

    train_labels = [train_labels[:, i] for i in range(7)]
    test_labels = [test_labels[:, i] for i in range(7)]

    # 创建数据增强的变换组合

    transform = A.Compose([
        # A.Rotate(limit=5, p=0.5),  # 限制较小的旋转角度
        # A.RandomScale(scale_limit=0.1),  # 限制较小的缩放范围
        A.ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=5,interpolation=0,border_mode=cv2.BORDER_CONSTANT, p=0.6),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1),  # 限制较小的亮度和对比度调整范围
        # A.GaussianBlur(blur_limit=3,p=0.5),  # 增加模糊程度，可自定义模糊核大小
        # A.MedianBlur(blur_limit=3, p=0.2),
        # A.ElasticTransform(p=0.5, alpha=15, sigma=120 * 0.05, alpha_affine=15 * 0.03),
        # A.GridDistortion(p=0.8),

        A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10),  # 限制较小的颜色偏移
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10),  # 限制较小的色相、饱和度和明度调整范围
        A.Resize(image_size[0], image_size[1]),
        # 添加其他所需的增强变换
    ])

    # 对图像进行数据增强
    augmented_train_images = []
    for train_image in train_images:
        augmented = transform(image=train_image)
        image = augmented["image"]
        augmented_train_images.append(image)

    augmented_train_images = np.array(augmented_train_images)
    train_images = augmented_train_images
    print("augmented_train_images:", augmented_train_images.shape, augmented_train_images.dtype)

    def show_augmented_result_demo(X_train, X_train_augmented):
        def visualize_segmentation(image, segmented_image):
            # 可视化分割结果
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            axes[0].imshow(image[:, :, [2, 1, 0]])
            axes[0].set_title('Original Image')

            axes[1].imshow(segmented_image[:, :, [2, 1, 0]])
            axes[1].set_title('Segmented Image')

            plt.show()

        # 选择样本进行测试
        index = 10
        sample_images = X_train[0:index]
        sample_images_augs = X_train_augmented[0:index]

        # 应用标签掩码来分割图像
        for sample_image, sample_images_aug in zip(sample_images, sample_images_augs):
            # 可视化分割结果
            visualize_segmentation(sample_image, sample_images_aug)

    show_augmented_result_demo(train_images, augmented_train_images)
    #
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
    model_save_path = "saved_models/cnn/my_cnn.h5"
    train(model_save_path)
