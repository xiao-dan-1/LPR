import os
import re
import cv2

import numpy as np
from matplotlib import pyplot as plt

from utils.LP_Correction import License_plate_correction
from utils.Progressbar import Progressbar

ccpd_path = "D:\Desktop\license plate recognition\CCPD\CCPD2019"
train_path = os.path.join(ccpd_path, r"splits\train.txt")


class CCPD:
    def __init__(self, path):
        self.path = path
        self.name = os.path.splitext(os.path.split(self.path)[-1])[0]
        self.width = 720
        self.height = 1160
        self.provincelist = [
            "皖", "沪", "津", "渝", "冀",
            "晋", "蒙", "辽", "吉", "黑",
            "苏", "浙", "京", "闽", "赣",
            "鲁", "豫", "鄂", "湘", "粤",
            "桂", "琼", "川", "贵", "云",
            "西", "陕", "甘", "青", "宁",
            "新"]
        self.wordlist = [
            "A", "B", "C", "D", "E",
            "F", "G", "H", "J", "K",
            "L", "M", "N", "P", "Q",
            "R", "S", "T", "U", "V",
            "W", "X", "Y", "Z", "0",
            "1", "2", "3", "4", "5",
            "6", "7", "8", "9"]

        self.template = "(?P<比例>\d+)-" \
                        "(?P<水平角度>\d+)_(?P<垂直角度>\d+)-" \
                        "(?P<左上角x>\d+)&(?P<左上角y>\d+)_(?P<右下角x>\d+)&(?P<右下角y>\d+)-" \
                        "(?P<右下x>\d+)&(?P<右下y>\d+)_(?P<左下x>\d+)&(?P<左下y>\d+)_(?P<左上x>\d+)&(?P<左上y>\d+)_(?P<右上x>\d+)&(?P<右上y>\d+)-" \
                        "(?P<省份>\d+)_(?P<地市>\d+)_(?P<车牌1>\d+)_(?P<车牌2>\d+)_(?P<车牌3>\d+)_(?P<车牌4>\d+)_(?P<车牌5>\d+)-" \
                        "(?P<亮度>\d+)-" \
                        "(?P<模糊度>\d+)"

        self.createInfo(self.name)

    def _get(self, s):
        return int(self.info.group(s))

    def createInfo(self, strings):
        # create info
        # print('creating info...')
        self.info = re.match(self.template, strings)
        self.Scale = self._get("比例")
        self.H_Angle = self._get("水平角度")
        self.V_Angle = self._get("垂直角度")
        self.LT = (self._get("左上x"), self._get("左上y"))
        self.RT = (self._get("右上x"), self._get("右上y"))
        self.LB = (self._get("左下x"), self._get("左下y"))
        self.RB = (self._get("右下x"), self._get("右下y"))
        self.LP = [self._get("省份"), self._get("地市"), self._get("车牌1"), self._get("车牌2"), self._get("车牌3"),
                   self._get("车牌4"),
                   self._get("车牌5")]
        self.Brightness = self._get("亮度")
        self.ambiguity = self._get("模糊度")
        # print('info created!')

    def getLP(self):
        return ''.join([self.provincelist[self.LP[0]]] + [self.wordlist[num] for num in self.LP[1:]])

    def CCPD_to_Mask(self, save_path=None):
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        polygon = np.array([self.LT, self.RT, self.RB, self.LB])  # [self.LT, self.RT, self.RB, self.LB]
        cv2.fillConvexPoly(mask, polygon, (255, 255, 255))  # 多边形内部填充
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        if save_path:
            if not os.path.exists(save_path):
                print("create save_path dir ")
                os.makedirs(save_path)
            cv2.imwrite(f"{save_path}/{self.name}.jpg", mask)
            # print(f"saved {save_path}/{self.name}.jpg")
        return mask

    def CCPD_to_LPdataset(self, save_path=None):
        img = cv2.imread(self.path)
        points = (self.LT, self.RT, self.LB, self.RB)
        lp = License_plate_correction(img=img, points=points)
        if save_path:
            if not os.path.exists(save_path):
                print(f"create {save_path} dir ")
                os.makedirs(save_path)
            # print(f"{save_path}/{self.getLP()}.jpg")
            cv2.imencode('.jpg', lp)[1].tofile(f"{save_path}/{self.getLP()}.jpg")
        return lp


def get_train_path():
    files_path = []
    with open(train_path, 'r') as f:
        for line in f.readlines():
            files_path.append(line.replace('\n', ''))
    return files_path


def CCPD2Mask(save_path=os.path.join(ccpd_path, "Mask")):
    files_path = get_train_path()
    show_num = 5
    for file_path in Progressbar(files_path[0:10]):
        file_path = os.path.join(ccpd_path, file_path)
        # print(file_path)
        ccpd = CCPD(file_path)
        mask = ccpd.CCPD_to_Mask(save_path)

        if show_num:
            img = cv2.imread(file_path)
            show_num = show_num - 1
            show_img = np.concatenate((img, mask, np.bitwise_and(img, mask)), axis=1)
            plt.imshow(show_img)
            plt.show()


def CCPD2LP(num=None, save_lp_path=os.path.join(ccpd_path, "lp")):
    files_path = get_train_path()
    show_num = 5
    for file_path in Progressbar(files_path[0:num], str="CCPD2LP"):
        file_path = os.path.join(ccpd_path, file_path)
        ccpd = CCPD(file_path)
        lp = ccpd.CCPD_to_LPdataset(save_lp_path)
        if show_num:
            show_num = show_num - 1
            img = cv2.imread(file_path)
            plt.subplot(1, 2, 1)
            plt.imshow(img[:, :, [2, 1, 0]])
            plt.subplot(1, 2, 2)
            plt.imshow(lp[:, :, [2, 1, 0]])
            plt.show()


# demo
if __name__ == '__main__':
    path = r'D:\Desktop\license plate recognition\CCPD\CCPD2019\ccpd_base\02-90_89-186&505_490&594-504&594_189&598_179' \
           r'&502_494&498-0_0_23_28_30_27_27-146-76.jpg'

    save_mask_path = "D:\Desktop\license plate recognition\CCPD\CCPD2019\mask"

    save_lp_path = "D:\Desktop\license plate recognition\CCPD\CCPD2019\lp"

    # # 基本信息Demo
    # ccpd = CCPD(path)
    # print(ccpd.Scale, ccpd.H_Angle, ccpd.V_Angle, ccpd.RB, ccpd.LB, ccpd.LT, ccpd.RT, ccpd.LP)
    # print(ccpd.getLP())

    # # CCPD 转 MASK Demo
    # ccpd = CCPD(path)
    # img = cv2.imread(path)
    # print("img:", img.shape, img.dtype)
    # Mask = ccpd.CCPD_to_Mask(save_path)
    # print("Mask:", Mask.shape, Mask.dtype)

    # CCPD 转 车牌数据 Demo
    ccpd = CCPD(path)
    lp = ccpd.CCPD_to_LPdataset(save_lp_path)
    cv2.imshow('lp', lp)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # cv2.imshow('img', np.concatenate((img, Mask), axis=1))
    # cv2.waitKey()
    # cv2.destroyAllWindows()
