import cv2
import numpy as np

# %%
img = cv2.imread(
    'labelme/data/01-86_91-298&341_449&414-458&394_308&410_304&357_454&341-0_0_14_28_24_26_29-124-24_json/label.png')
print(img)
label = img / np.max(img[:, :, 2]) * 255  # 提高亮度
label[:, :, 0] = label[:, :, 1] = label[:, :, 2]  #
print(np.max(label[:, :, 2]))
print(set(label.flatten()))  # set()创建一个无序不重复元素集 .ravel()拉成一维数组 修改会影响到原本数组 flatten() 修改不会影响到原本数组
label1 = label.ravel()
print(label.shape)
print(label1.shape)
