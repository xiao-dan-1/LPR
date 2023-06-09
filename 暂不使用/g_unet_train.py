# 导入模型定义和训练函数
from g_unet_class import UNet
from utils.LoadDatasets import *

# 定义数据集路径和一些训练参数

# data_path = 'datasets/label-studio'
data_path = "../datasets/CCPD"
image_folder = 'Images'
mask_folder = 'Mask'

image_size = (512, 512)
validation_split = 0.2

batch_size = 8
num_epochs = 300

save_model_path = "../saved_models/unet/gunet.h5"

# 创建模型
input_shape = (512, 512, 3)
unet = UNet(input_shape, depth=4, filters=8)

# 加载数据集
X_train, X_test, y_train, y_test = unet.load_data(data_path, image_folder, mask_folder, image_size,
                                                  test_size=validation_split, random_state=42)
print(X_train.shape, X_train.dtype)
print(y_train.shape, y_train.dtype)
print(X_test.shape, X_test.dtype)
print(y_test.shape, y_test.dtype)
# 选择一个样本进行测试
index = 5
sample_images = X_train[0:index]
sample_masks = y_train[0:index]

for sample_image, sample_mask in zip(sample_images, sample_masks):
    # 应用标签掩码来分割图像
    segmented_image = apply_mask(sample_image, sample_mask)
    # 可视化分割结果
    visualize_segmentation(sample_image, sample_mask, segmented_image)

# 描述网络结构
unet.describe_model()

# # 进行模型训练
unet.train(X_train, X_test, y_train, y_test, batch_size, num_epochs)

# 保存模型
unet.save_model(save_model_path)

#
unet.show_history()
