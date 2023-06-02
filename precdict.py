import matplotlib.pyplot as plt

from utils.LoadDataset import _Loadimgs
from tensorflow.keras import models
from utils.LP_Correction import *


def UNet_Precdict(model_path, img_path):
    # 加载图片
    imgs = _Loadimgs(img_path)
    print(len(imgs), imgs.shape)

    # 加载模型
    unet = models.load_model(model_path)
    imgs_mask = unet.predict(imgs)
    return imgs, imgs_mask.astype(np.uint8)


def lp_correction(imgs, imgs_mask):
    lp = []
    for img, img_mask in zip(imgs, imgs_mask):
        cv2.imwrite("img.jpg", img)
        cv2.imwrite("img_mask.jpg", img_mask)
        points = get_points_from_mask(img_mask)
        for c in points:
            print(c)
            cv2.circle(img_mask, c, radius=2, color=(0, 0, 255), thickness=-1)
        # cv_show(img_mask)
        dst = License_plate_correction(img, points)
        lp.append(dst)
    return np.array(lp)


def result_show(imgs, imgs_mask, lps, out_path=None):
    i = 0
    for img, img_mask, lp in zip(imgs, imgs_mask, lps):
        i = i + 1
        # print(img.shape, img.dtype, img_mask.shape, img_mask.dtype)
        if out_path:
            cv2.imwrite(f"{out_path}/{i}.jpg", np.concatenate((img, img_mask, np.bitwise_and(img, img_mask)), axis=1))
        if i >= 10:
            continue
        plt.subplot(2, 1, 1)
        plt.imshow(np.concatenate((img, img_mask, np.bitwise_and(img, img_mask)), axis=1)[:, :, [2, 1, 0]])
        plt.subplot(2, 1, 2)
        plt.imshow(lp[:, :, [2, 1, 0]])
        plt.show()


if __name__ == "__main__":
    img_path = "datasets/label-studio/images/"
    model_path = "saved_models/unet.h5"
    out_path = "result"
    # 预测分割
    imgs, imgs_mask = UNet_Precdict(model_path, img_path)

    # 车牌校正
    lps = lp_correction(imgs, imgs_mask)

    # 展示
    result_show(imgs, imgs_mask, lps)

