import os, sys

dir_mytest = "/"
sys.path.insert(0, dir_mytest)
import cv2
import numpy as np
from utils.Progressbar import Progressbar

path = r"/datasets/CCPD/new_lps"
save_path = r"/datasets/CCPD/good_lps"
if not os.path.exists(save_path):
    os.makedirs(save_path)
# 1.
# 29642:23306
h_mean, h_std, h_var = 36.27476014202614, 9.045943427611023, 81.82909249553907
w_mean, w_std, w_var = 161.6702425020775, 39.94220829730877, 1595.3800036656014
w_h_mean, w_h_std, w_h_var = 4.508401689011867, 0.8019373039447997, 0.6431034394582541

# # 2.
# # 39903 :13045
# h_mean, h_std, h_var = 34.358019394147426, 12.29413798551853, 151.1458288069696
# w_mean, w_std, w_var = 152.9679910752596, 55.3247932859997, 3060.8327521385972
# w_h_mean, w_h_std, w_h_var = 4.538653640738838, 1.1330521813354621, 1.283807245629049

h_th = [h_mean - h_std, h_mean + h_std]
w_th = [w_mean - w_std, w_mean + w_std]
w_h_th = [w_h_mean - w_h_std, w_h_mean + w_h_std]


def func(x, th):
    return th[0] < x < th[1]


print(h_th)
print(func(28, h_th))

files_name = os.listdir(path)
heights = []
widths = []
w_hs = []
i = 0
j = 0
for file_name in Progressbar(files_name):
    img_path = os.path.join(path, file_name)
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    height, width, t = img.shape
    w_h = width / height

    if func(height, h_th) and func(width, w_th) and func(w_h, w_h_th):
        cv2.imencode('.jpg', img)[1].tofile(f"{save_path}/{file_name}")
        # cv2.imshow('img', img)
        # cv2.waitKey()
        i += 1
    else:

        heights.append(height)
        widths.append(width)
        w_hs.append(w_h)
        j += 1

print(i)
print(j)

# print(heights, widths, w_hs)

heights = np.array(heights)
widths = np.array(widths)
w_hs = np.array(w_hs)

h_mean, h_std, h_var = np.mean(heights), np.std(heights), np.var(heights)
w_mean, w_std, w_var = np.mean(widths), np.std(widths), np.var(widths)
w_h_mean, w_h_std, w_h_var = np.mean(w_hs), np.std(w_hs), np.var(w_hs)
print("mean----std-----var")
print(h_mean, h_std, h_var)
print(w_mean, w_std, w_var)
print(w_h_mean, w_h_std, w_h_var)
