import cv2
import matplotlib.pyplot as plt
import numpy as np


def cv_show(name, image):
    cv2.imshow(name, image)
    cv2.waitKey()
    cv2.destroyWindow(name)


def plt_gray_show(img):
    plt.imshow(img, cmap='gray')
    plt.show()


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
    cv2.createTrackbar('ksize', 'TrackBars', 1, 9, Threshold_analyzer)

    # 设置默认值
    cv2.setTrackbarPos('lowThreshold', 'TrackBars', 0)
    cv2.setTrackbarPos('maxThreshold', 'TrackBars', 0)

    cv2.waitKey(0)


def GaussianBlurThreshold(ksize):
    try:
        img_gaussianBlur = cv2.GaussianBlur(gray_img, (ksize, ksize), 0)
        cv2.imshow('TrackBars', img_gaussianBlur)
    except:
        print(f"ERROR:ksize!={ksize}")


def CannyThreshold(lowThreshold, maxThreshold):
    kernel_size = 3
    edges = cv2.GaussianBlur(gray_img, (5, 5), 0)
    edges = cv2.Canny(edges,
                      lowThreshold,
                      maxThreshold,
                      apertureSize=kernel_size)
    # dst = cv2.bitwise_and(img, img, mask=detected_edges)  # just add some colours to edges from original image.
    cv2.imshow('TrackBars', edges)


# 字符分割

# 字符识别

def find_lp(arrays):
    # print(arrays[0], type(arrays[0]))
    # print(arrays[0][:,:,0])
    for i, array in enumerate(arrays):
        array = np.squeeze(array)  # 降维
        x = array[:, 0]
        y = array[:, 1]
        # x_min, x_max = min(x), max(x)
        # y_min, y_max = min(y), max(y)

        # 排序
        array = np.sort(array, 0)
        # print(array)  # 0:按列排列
        # 绘制散点
        plt.scatter(array[:, 0], array[:, 1], c='r')
        plt.show()
        array = array[np.newaxis, :]
        print(array)
        imgc = cv2.drawContours(img.copy(), [arrays[i]], -1, (0, 0, 255))
        cv_show('imgc', imgc)

        # for x in range(x_min, y_max + 1):
        #     print()


template = np.load(file="template.npy")

if __name__ == "__main__":
    # 加载数据
    img = cv2.imread('datasets/test.jpg')
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # create_Threshold_analyzer()
    # 图像预处理
    img_gaussianBlur = cv2.GaussianBlur(gray_img, (3, 3), 0)

    # 边缘检测
    edges = cv2.Canny(img_gaussianBlur, threshold1=57, threshold2=400, apertureSize=3)

    # 开运算
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel=kernel)

    # edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1)), iterations=1)
    edges = cv2.erode(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5)), iterations=1)

    # 腐蚀-膨胀

    # edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 10)), iterations=1)
    # edges = cv2.erode(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1)), iterations=1)
    #
    cv_show('edges', edges)
    # # 中值滤波
    img_filter = cv2.medianBlur(edges, 5)
    # img_filter = cv2.blur(edges, (5, 5))
    # 腐蚀噪点
    img_filter = cv2.erode(img_filter, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5)), iterations=1)

    # cv_show('edges', img_filter)
    #
    # #
    # 找边界
    contours = cv2.findContours(img_filter, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    imgc = cv2.drawContours(img.copy(), contours[1], -1, (0, 0, 255))
    # cv_show('imgc', imgc)

    # 特征提取
    retval = [cv2.moments(contours) for contours in contours[1]]
    # print(retval)
    print(retval[0])

    # # 统计学分析
    Areas = [cv2.contourArea(countour) for countour in contours[1]]
    print(Areas)
    new_contours = []
    for countour in contours[1]:
        Area = cv2.contourArea(countour)
        if Area < np.mean(Areas):
            continue
        else:
            new_contours.append(countour)
    new_contours = np.array(new_contours)
    print(new_contours)
    imgc = cv2.drawContours(img.copy(), new_contours, -1, (0, 0, 255))
    cv_show('imgc', imgc)

    #
    for contour in new_contours:
        ret = cv2.matchShapes(contour, template, 1, 0.0)
        print(ret)
        if ret < 0.7:
            imgc = cv2.drawContours(img.copy(), [contour], -1, (0, 255, 0))
            cv_show('imgc', imgc)

    # imgc = cv2.drawContours(img.copy(), contours[1], -1, (0, 0, 255))
    # cv_show('edges',imgc)
    # rects = []
    #
    # for contour in contours[1]:
    #     rect = cv2.boundingRect(contour)
    #     rects.append(rect)
    #     # print(rect)
    #     # cv2.rectangle(imgr, rect[0:2], (rect[0] + rect[2], rect[1] + rect[3]), color=(255, 0, 0), thickness=2)
    #
    # imgr = img.copy()
    # for rect in rects:
    #     x, y, w, h = rect
    #     w_h = w / h
    #     if w_h < 4:
    #         continue
    #     else:
    #         cv2.rectangle(imgr, (x, y), (x+w, y+h), color=(255, 0, 0), thickness=2)
    #
    # # img = cv2.drawContours(img.copy(), rect, -1, (0, 0, 255), 3)
    # cv_show('imgr', imgr)
    # edges = CannyThreshold(100)
    # contours= cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # contours = imutils.grab_contours(contours)
    # contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    #
    # print(contours)
    # cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
