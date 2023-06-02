import cv2
from utils.LP_Correction import *
import time


def CannyThreshold(lowThreshold, maxThreshold):
    kernel_size = 3
    edges = cv2.Canny(img_mask,
                      lowThreshold,
                      maxThreshold,
                      apertureSize=kernel_size)
    # dst = cv2.bitwise_and(img, img, mask=detected_edges)  # just add some colours to edges from original image.
    cv2.imshow('TrackBars', edges)


def Threshold_analyzer(x):
    # 获取滑动条参数
    lowThreshold = cv2.getTrackbarPos("lowThreshold", "TrackBars")
    maxThreshold = cv2.getTrackbarPos("maxThreshold", "TrackBars")
    # ksize = cv2.getTrackbarPos("ksize", "TrackBars")
    # print(lowThreshold, maxThreshold)
    #
    # GaussianBlurThreshold(ksize)
    CannyThreshold(lowThreshold, maxThreshold)


def create_Threshold_analyzer():
    # 创建窗口
    cv2.namedWindow('TrackBars')
    cv2.createTrackbar('lowThreshold', 'TrackBars', 0, 500, Threshold_analyzer)
    cv2.createTrackbar('maxThreshold', 'TrackBars', 0, 800, Threshold_analyzer)
    # 设置默认值
    cv2.setTrackbarPos('lowThreshold', 'TrackBars', 0)
    cv2.setTrackbarPos('maxThreshold', 'TrackBars', 0)

    cv2.waitKey(0)


# deom
if __name__ == '__main__':
    img_mask = cv2.imread('mask.jpg')
    # cv_show(img_mask)
    time_start = time.clock()  # 记录开始时间
    # create_Threshold_analyzer()

    points = get_points_from_mask(img_mask)
    # print(LT, type(LT))

    for c in points:
        print(c)
        cv2.circle(img_mask, c, radius=2, color=(0, 0, 255), thickness=-1)

    cv_show(img_mask)

    dst = License_plate_correction(img_mask, points)
    print(dst.shape, dst.dtype)
    cv2.imshow('dst', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    time_end = time.clock()  # 记录结束时间
    time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
    print("time_sum:", time_sum)
