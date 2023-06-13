import tensorflow as tf

from datasets.CBLPRD.LoadDatasets import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
plt.rcParams['font.sans-serif'] = ['SimHei']

characters = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫",
              "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "0", "1", "2",
              "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M",
              "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]


def cnn_predict(cnn, imgs):
    # print(imgs.shape)
    preds = cnn.predict(imgs)  # 预测形状应为(1,80,240,3)
    # print(preds)
    # lps = [[np.argmax(pre) for pre in pred] for pred in preds]
    images = []
    pre_lps = []
    probabilitys = []
    for pred in preds:
        lp = []
        probability = []
        # print(sum([sum(pre > 0.7) for pre in pred]))
        for pre in pred:
            max_index = np.argmax(pre)
            max_probability = pre[max_index]
            lp.append(max_index)
            probability.append(max_probability)
        pre_lps.append(lp)
        probabilitys.append(probability)

    pre_lps = np.array(pre_lps, dtype=np.uint8).T
    pre_lps = [lpdecode(pre_lp) for pre_lp in pre_lps]
    probabilitys = np.array(probabilitys, dtype=np.float16).T

    return pre_lps, probabilitys


# 获取车牌
def get_lp(imgs, masks):
    lps = []
    labels = []
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
        # 利用最小外接矩阵获取车牌
        img_rots = []
        mask_rots = []
        for cont in contours:
            dst = img.copy()
            # 外接矩形
            x, y, w, h = cv2.boundingRect(cont)
            cv2.rectangle(dst, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # 对每个轮廓点求最小外接矩形
            center, size, angle = cv2.minAreaRect(cont)  # 返回(x,y)(w,h)旋转角度
            # print("rect:", center, size, angle)
            box = cv2.boxPoints((center, size, angle))  # cv2.boxPoints可以将轮廓点转换为四个角点坐标
            # startidx = box.sum(axis=1).argmin() # 这一步不影响后面的画图，但是可以保证四个角点坐标为顺时针
            # box = np.roll(box, 4 - startidx, 0)
            # 在原图上画出预测的外接矩形
            box = box.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(dst, [box], True, (0, 255, 0), 2)
            # plt_show(dst)

            angle = angle if angle < 45 else angle - 90
            M = cv2.getRotationMatrix2D(center, angle, scale=1)  # （center,angle,scale）计算旋转矩阵
            img_rot = cv2.warpAffine(img.copy(), M, (img.shape[0], img.shape[1]))  # 仿射变换
            mask_rot = cv2.warpAffine(mask.copy(), M, (img.shape[0], img.shape[1]))  # 仿射变换

            size = sorted(list(map(int, size)), reverse=True)

            img_rot = cv2.getRectSubPix(img_rot, size, center)  # 获取矩形
            mask_rot = cv2.getRectSubPix(mask_rot, size, center)  # 获取矩形
            # cv_show(img_rot)
            # cv_show(mask_rot)
            img_rots.append(img_rot)
            mask_rots.append(mask_rot)
            # Areas = cv2.contourArea(cont)
            # templates = "Areas\t{}\tw_h\t{:.4f}\tAreas_w\t{:.4f}\tAreas_h\t{:.4f}\tsize\t{}"
            # result  = templates.format(Areas,  size[0] / size[1], Areas / size[0], Areas / size[1],size)
            # print(result)
        lps.append(img_rots)
        labels.append(mask_rots)
    return lps, labels


def result_show(imgs, pres, probabilitys, save_path=None):
    show_num = 30
    for img, pre, probability in zip(imgs, pres, probabilitys):
        print(f"pre:{pre}\tacc:{probability}")
        acc_num = np.sum(probability > 0.9)
        if acc_num < 5:
            print("不是车牌！！！")
            continue
        if save_path:
            if not os.path.exists(save_path):
                print(f"create {save_path} dir ")
                os.makedirs(save_path)
            cv2.imencode('.jpg', img)[1].tofile(f"{save_path}/{''.join(pre)}.jpg")
        if show_num:
            show_num = show_num - 1
            plt.imshow(img[:, :, [2, 1, 0]])
            plt.title(f"{pre}")
            plt.show()


if __name__ == '__main__':
    out_path = "result/cnn"
    # 加载图片
    # img_path = "D:/Desktop/license plate recognition/CCPD/CCPD2019/lp/"
    # data_images, data_labels = LoadDataset_for_CNN(img_path, 500)

    # datasets_path = "datasets/lp"
    datasets_path = "../datasets/CBLPRD"
    _, data_images, _, test_labels = LoadData(datasets_path, 0.5, 10)
    # 加载模型
    model_path = "saved_models/cnn/best_48_from_goodlps.h5"
    model = tf.keras.models.load_model(model_path)
    pres, probabilitys = cnn_predict(model, data_images)
    print(pres)
    print(probabilitys)
    # print(probabilitys > 0.6)
    # print(np.sum(probabilitys > 0.6, axis=1))
    # 展示
    result_show(data_images, pres, probabilitys)
