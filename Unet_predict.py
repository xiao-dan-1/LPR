import os.path

import matplotlib.pyplot as plt
import tensorflow as tf
from utils.LoadDatasets import *
from datasets.CCPD.CCPD import CCPD
from utils.LP_Correction import *
from utils.Progressbar import Progressbar


# 评估模型
def evaluate_model_with_visualization(model, X_test):
    # 进行预测
    y_preds = model.predict(X_test)
    y_preds = np.squeeze(y_preds, axis=-1)  #
    y_preds = np.round(y_preds).astype(np.uint8)

    for sample_image, sample_mask in zip(X_test, y_preds):
        # 应用标签掩码来分割图像
        segmented_image = apply_mask(sample_image, sample_mask)
        # 可视化分割结果
        visualize_segmentation(sample_image, sample_mask, segmented_image)


# 车牌旋转校正
def lp_correction(imgs, imgs_mask):
    lp = []
    for img, img_mask in zip(imgs, imgs_mask):
        # cv2.imwrite("img.jpg", img)
        # cv2.imwrite("img_mask.jpg", img_mask)
        points = get_points_from_mask(img_mask)
        print(points, type(points))
        show_img = img.copy()
        for c in points:
            cv2.circle(show_img, c, radius=2, color=(0, 0, 255), thickness=-1)
        plt_show(show_img)
        dst = License_plate_correction(img, points)
        lp.append(dst)
    return np.array(lp)


if __name__ == '__main__':
    # 基本参数
    model_path = "saved_models/unet/65-gunet.h5"
    data_path = "../CCPD/my_use"  # ,"datasets/label-studio"
    image_folder = 'Images'
    images_path = os.path.join(data_path, image_folder)
    input_shape = (512, 512, 3)
    image_size = (512, 512)
    validation_split = 0.2

    # 加载数据集
    datas = LoadData_for_predict(images_path, image_size, 50)
    print(datas.shape, datas.dtype)

    # 加载模型
    model = tf.keras.models.load_model(model_path)

    # 预测
    masks = model.predict(datas)
    masks = np.squeeze(masks, axis=-1)  # 降维 去最后一维
    masks = np.round(masks).astype(np.uint8)  # 二值化，数据类型转换

    # 获取车牌
    cor_lps, cor_masks = get_lp(datas, masks)
    # 显示
    i = 0
    for image, mask, cor_lp, cor_mask in Progressbar(zip(datas, masks, cor_lps, cor_masks)):
        if np.max(mask)==0: print("kong")
        i += 1
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        save_path = "result/unet_predict"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        lp_mask = cv2.cvtColor(cor_mask, cv2.COLOR_GRAY2BGR)
        # print("cor_lp:", cor_lp.shape, cor_lp.dtype)
        # print("lp_mask:", lp_mask.shape, lp_mask.dtype)
        try:
            plt.subplot(2, 1, 1)
            plt.axis("off")  # 取消坐标轴
            plt.title("car")
            plt.imshow(np.concatenate((image, mask * 255), axis=1))
            plt.subplot(2, 1, 2)
            plt.axis("off")  # 取消坐标轴
            plt.title("mask")
            plt.imshow(np.concatenate((cor_lp, lp_mask), axis=1)[:, :, [2, 1, 0]])
            plt.savefig(f"{save_path}/{i}.jpg")
            plt.show()
        except:
            plt.savefig(f"{save_path}/{i}.jpg")
            print('显示失败')
