import os
import re
import cv2

import numpy as np


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


# demo
if __name__ == '__main__':
    path = r'D:\Desktop\license plate recognition\CCPD\CCPD2019\ccpd_base\02-90_89-186&505_490&594-504&594_189&598_179' \
           r'&502_494&498-0_0_23_28_30_27_27-146-76.jpg'

    save_path = "D:\Desktop\license plate recognition\CCPD\CCPD2019\mask"

    ccpd = CCPD(path)
    print(ccpd.Scale, ccpd.H_Angle, ccpd.V_Angle, ccpd.RB, ccpd.LB, ccpd.LT, ccpd.RT, ccpd.LP)
    print(ccpd.getLP())

    img = cv2.imread(path)
    print("img:", img.shape, img.dtype)
    mask = ccpd.CCPD_to_Mask(save_path)
    print("mask:", mask.shape, mask.dtype)

    # cv2.imshow('img', np.concatenate((img, mask), axis=1))
    # cv2.waitKey()
    # cv2.destroyAllWindows()
