# load a image
import os.path
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np


# from utils.CCPD import CCPD

def plt_show(img):
    plt.imshow(img)
    plt.show()


def cv_show(img):
    cv2.imshow(f'{img}', img)
    cv2.waitKey()
    cv2.destroyAllWindows()


# 线性规划
def linear_program(contours, func):
    # print("contours.shape:", contours)  # (46,1,2)
    max_point = np.array([0, 0])
    min_point = np.array([512, 512])
    print()
    for contour in contours:
        for c in contour:
            c = np.squeeze(c)
            z1 = func(c)
            if func(min_point) < z1 < func(max_point):
                continue
            elif z1 < func(min_point):
                min_point = c
            elif func(min_point) < z1:
                max_point = c
    return min_point.tolist(), max_point.tolist()


def get_points_from_mask(img_mask):
    if np.max(img_mask) == 1:
        img_mask = img_mask * 255
    # 开运算
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    img_mask = cv2.morphologyEx(img_mask, cv2.MORPH_CLOSE, kernel=kernel)
    # 边缘检测
    edges = cv2.Canny(img_mask, threshold1=0, threshold2=500, apertureSize=3)  # time_sum 0.009975194931030273

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    imgc = cv2.drawContours(np.zeros_like(edges), contours, -1, 255, thickness=5)
    plt_show(imgc)
    LT, RB = linear_program(contours, lambda x: x.tolist()[0] + x.tolist()[1])
    LB, RT = linear_program(contours, lambda x: x.tolist()[0] - x.tolist()[1])
    print()
    return LT, RT, LB, RB  # 返回坐标列表


# 中国汽车尺寸为 440mm×140mm
def License_plate_correction(img, points, out_size=(220, 70)):
    pts_o = np.float32(points)  # [左上，右上，左下，右下]原图位置
    pts_d = np.float32([[0, 0], [220, 0], [0, 70], [220, 70]])  # 变换后的位置
    # 映射变换
    # 计算映射矩阵
    M = cv2.getPerspectiveTransform(pts_o, pts_d)
    # apply transformation
    dst = cv2.warpPerspective(img, M, out_size)  #
    return dst


# 获取车牌
def get_lp(imgs, masks):
    cor_lps = []
    cor_masks = []
    for img, mask in zip(imgs, masks):
        if np.max(mask) == 1:
            mask = mask * 255
        mask = cv2.medianBlur(mask, 5)
        # 开运算
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel=kernel)
        # 将图片二值化
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        # 在二值图上寻找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        # print(contours)
        Area = [cv2.contourArea(cont) for cont in contours]
        # print("Area:",Area)
        maxArea_index = np.argmax(np.array([cv2.contourArea(cont) for cont in contours]))
        # print(maxArea_index)
        # 考虑图像中只有一张车牌，选取面积最大的作为车牌
        contours = [contours[maxArea_index]]
        # 利用最小外接矩阵获取车牌
        img_rots = []
        mask_rots = []
        for cont in contours:
            dst = img.copy()
            # 外接矩形
            x, y, w, h = cv2.boundingRect(cont)
            cv2.rectangle(dst, (x, y), (x + w, y + h), (0, 0, 255), 2)
            # 最小外接矩形
            center, size, angle = cv2.minAreaRect(cont)  # 返回(x,y)(w,h)旋转角度
            # print("rect:", center, size, angle)
            box = cv2.boxPoints((center, size, angle))  # cv2.boxPoints可以将轮廓点转换为四个角点坐标
            # 在原图上画出预测的外接矩形
            box = box.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(dst, [box], True, (0, 255, 0), 2)
            # plt_show(dst)

            # 旋转校正
            angle = angle if angle < 45 else angle - 90
            M = cv2.getRotationMatrix2D(center, angle, scale=1)  # （center,angle,scale）计算旋转矩阵
            img_rot = cv2.warpAffine(img.copy(), M, (img.shape[0], img.shape[1]))  # 仿射变换
            mask_rot = cv2.warpAffine(mask.copy(), M, (img.shape[0], img.shape[1]))  # 仿射变换
            size = sorted(list(map(int, size)), reverse=True)
            img_rot = cv2.getRectSubPix(img_rot, size, center)  # 获取矩形
            mask_rot = cv2.getRectSubPix(mask_rot, size, center)  # 获取矩形
            # cv_show(img_rot)
            # cv_show(mask_rot)
            cor_lps.append(img_rot)
            cor_masks.append(mask_rot)
            # Areas = cv2.contourArea(cont)
            # template = "Areas\t{}\tw_h\t{:.4f}\tAreas_w\t{:.4f}\tAreas_h\t{:.4f}\tsize\t{}"
            # result  = template.format(Areas,  size[0] / size[1], Areas / size[0], Areas / size[1],size)
            # print(result)
        # cor_lps.append(img_rots)
        # cor_masks.append(mask_rots)
    return cor_lps, cor_masks


if __name__ == '__main__':
    pass
    # # path = "D:/Desktop/license plate recognition/CCPD/CCPD2019/ccpd_base/"
    # # files_path = get_files_path(path)
    # ccpd_path = "D:\Desktop\license plate recognition\CCPD\CCPD2019"
    #
    # train_path = os.path.join(ccpd_path, r"splits\train.txt")
    # files_path = []
    # with open(train_path, 'r') as f:
    #     for line in f.readlines():
    #         files_path.append(line.replace('\n', ''))
    # print(files_path, len(files_path))
    # for file_path in files_path[50:60]:
    #     ccpd = CCPD(os.path.join(ccpd_path, file_path))
    #     img = cv2.imread(ccpd.path)
    #     points = [ccpd.LT, ccpd.RT, ccpd.LB, ccpd.RB]
    #
    #     dst = License_plate_correction(img, points)
    #
    #     print(dst.shape, dst.dtype)
    #     cv2.imshow('img', img)
    #     cv2.imshow('dst', dst)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

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
