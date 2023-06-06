import os
import cv2
import albumentations as A

image_folder = '../datasets/CCPD/Images'
mask_folder = '../datasets/CCPD/Mask'

augmented_image_folder = "../datasets/aug/Images"
augmented_mask_folder = "../datasets/aug/Mask"
if not os.path.exists(augmented_image_folder):
    os.makedirs(augmented_image_folder)
if not os.path.exists(augmented_mask_folder):
    os.makedirs(augmented_mask_folder)

target_shape = (512, 512, 3)


def apply_augmentation(image, mask):
    # 定义图像增强的变换
    transform = A.Compose([
        A.Resize(target_shape[0], target_shape[1]),
        A.HorizontalFlip(p=0.5),  # 水平翻转，概率为0.5
        A.Rotate(limit=10, p=0.5),  # 随机旋转，角度范围为-10到10，概率为0.5
        A.RandomBrightnessContrast(p=0.2),  # 随机调整亮度和对比度，概率为0.2
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),  # 高斯模糊，模糊程度范围为3到7，概率为0.3
        # 添加其他所需的增强变换
    ])

    # 对图像和掩码应用增强变换
    augmented = transform(image=image, mask=mask)
    augmented_image = augmented["image"]
    augmented_mask = augmented["mask"]

    return augmented_image, augmented_mask


# 获取图像和掩码文件列表
image_files = sorted(os.listdir(image_folder))[0:5]
mask_files = sorted(os.listdir(mask_folder))[0:5]

# 遍历每个图像和掩码文件
for image_file, mask_file in zip(image_files, mask_files):
    # 读取图像和掩码
    image_path = os.path.join(image_folder, image_file)
    mask_path = os.path.join(mask_folder, mask_file)
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path)

    # 应用数据增强
    augmented_image, augmented_mask = apply_augmentation(image, mask)

    # 保存增强后的图像和掩码
    augmented_image_path = os.path.join(augmented_image_folder, image_file)
    augmented_mask_path = os.path.join(augmented_mask_folder, mask_file)
    cv2.imwrite(augmented_image_path, augmented_image)
    cv2.imwrite(augmented_mask_path, augmented_mask)
