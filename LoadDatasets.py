import os
import cv2
import numpy as np

np.set_printoptions(threshold=np.inf)


def cv_show(name, image):
    cv2.imshow(name, image)
    cv2.waitKey()
    cv2.destroyWindow(name)


# 输入文件夹，返回文件路径列表
def get_files_path(path):
    files_path = []
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            file_path = os.path.join(root, filename)
            files_path.append(file_path)
    return files_path


# 将json标签转换成数据集
def labelme_json_to_dataset(in_path, out_path):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    files_path = get_files_path(in_path)
    for file_path in files_path:
        root, file_name = os.path.split(file_path)
        os.system(f'labelme_json_to_dataset "{file_path}" -o "{out_path}/{os.path.splitext(file_name)[0]}_json"')


#
def dataset_to_trainset(in_path, train_images_path, train_labels_path):
    if not os.path.exists(train_images_path):
        os.makedirs(train_images_path)
    if not os.path.exists(train_labels_path):
        os.makedirs(train_labels_path)
    files_path = get_files_path(in_path)
    for file_path in files_path:
        root, file_name = os.path.split(file_path)
        if 'img.png' == file_name:
            img = cv2.imread(file_path)
            cv2.imwrite(train_images_path + f'{os.path.split(root)[-1]}.png', img)
        elif 'label.png' == file_name:
            label = cv2.imread(file_path)
            label = label / np.max(label[:, :, 2]) * 255  # 提高亮度
            label[:, :, 0] = label[:, :, 1] = label[:, :, 2]  #
            print(np.max(label[:, :, 2]))
            print(set(label.ravel()))  # set()创建一个无序不重复元素集 .ravel()拉成一维数组 修改会影响到原本数组 flatten() 修改不会影响到原本数组
            cv2.imwrite(train_labels_path + f'{os.path.split(root)[-1]}.png', label)


if __name__ == "__main__":
    # 数据转换
    in_path = 'labelme/json'
    out_path = 'labelme/data'
    # labelme_json_to_dataset(in_path, out_path)

    # 提取image和label
    train_images_path_path = 'labelme/train_images/'
    train_labels_path_path = 'labelme/train_labels/'
    dataset_to_trainset(out_path, train_images_path_path, train_labels_path_path)
