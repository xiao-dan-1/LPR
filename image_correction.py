# load a image
import os.path
import re

import cv2
import numpy as np
from utils.CCPD import CCPD
from utils.Json2datasets import get_files_path


def transformation(ccpd):
    img = cv2.imread(ccpd.path)
    pts_o = np.float32([ccpd.LT, ccpd.RT, ccpd.LB, ccpd.RB])  # [左上，右上，左下，右下]这四个点为原始图片上数独的位置
    pts_d = np.float32([[0, 0], [220, 0], [0, 70], [220, 70]])  # 这是变换之后的图上四个点的位置
    # 映射变换
    # 计算映射矩阵
    M = cv2.getPerspectiveTransform(pts_o, pts_d)
    # apply transformation
    dst = cv2.warpPerspective(img, M, (220, 70))  # 最后一参数是输出dst的尺寸。可以和原来图片尺寸不一致。按需求来确定
    print(dst.shape, dst.dtype)
    cv2.imshow('img', img)
    cv2.imshow('dst', dst)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return dst


if __name__ == '__main__':
    # path = "D:/Desktop/license plate recognition/CCPD/CCPD2019/ccpd_base/"
    # files_path = get_files_path(path)
    ccpd_path = "D:\Desktop\license plate recognition\CCPD\CCPD2019"

    train_path = os.path.join(ccpd_path, r"splits\train.txt")
    files_path = []
    with open(train_path, 'r') as f:
        for line in f.readlines():
            files_path.append(line.replace('\n', ''))
    print(files_path, len(files_path))
    for file_path in files_path[10:20]:
        ccpd = CCPD(os.path.join(ccpd_path, file_path))
        transformation(ccpd)

    # file_path = r'D:\Desktop\license plate recognition\CCPD\CCPD2019\ccpd_base\02-90_89-186&505_490&594-504&594_189&598_179' \
    #        r'&502_494&498-0_0_23_28_30_27_27-146-76.jpg'
    # img = cv2.imread(file_path)
    # rows, cols = img.shape[:2]
    #
    # ccpd = CCPD(file_path)
    # # original pts
    # pts_o = np.float32([ccpd.LT, ccpd.RT, ccpd.LB, ccpd.RB])  # [左上，右上，左下，右下]这四个点为原始图片上数独的位置
    # pts_d = np.float32([[0, 0], [300, 0], [0, 100], [300, 100]])  # 这是变换之后的图上四个点的位置
    #
    # # 映射变换
    # # 计算映射矩阵
    # M = cv2.getPerspectiveTransform(pts_o, pts_d)
    # # apply transformation
    # dst = cv2.warpPerspective(img, M, (300, 100))  # 最后一参数是输出dst的尺寸。可以和原来图片尺寸不一致。按需求来确定
    #
    # cv2.imshow('img', img)
    # cv2.imshow('dst', dst)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
