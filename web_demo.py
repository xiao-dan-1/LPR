import cv2
import gradio as gr
from PIL import Image
import os.path

import matplotlib.pyplot as plt
import tensorflow as tf
from utils.LoadDatasets import *
from datasets.CCPD.CCPD import CCPD
from utils.LP_Correction import *
from utils.Progressbar import Progressbar

model_path = "saved_models/unet/65-gunet.h5"
# 加载unet,获得车牌
model = tf.keras.models.load_model(model_path)


def get_mask(image):
    image_size = (512, 512)
    image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), image_size).astype(np.uint8)
    # 预测
    masks = model.predict(np.expand_dims(image, axis=0))
    mask = np.where(np.squeeze(masks) > 0.5, 255, 0).astype(np.uint8)  # 二值化，数据类型转换

    return mask


# 0.95687395
# 255
if __name__ == '__main__':
    interface = gr.Interface(fn=get_mask, inputs="image", outputs="image")
    interface.launch()
    # # 加载数据集
    # data_path = "../CCPD/my_use"  # "../CCPD/my_use",
    # image_folder = 'Images'
    # image = cv2.imread(os.path.join(data_path, image_folder,
    #                                 '00224137931034-90_87-351&564_451&606-440&599_362&600_359&572_437&571-0_0_3_21_30_28_24-88-5.jpg'))
    # mask = get_LP(image)
    # plt.imshow(mask)
    # plt.show()
