import cv2
import gradio as gr
import os.path
import matplotlib.pyplot as plt
import tensorflow as tf
from utils.LP_Correction import *
from CNN_predict import cnn_predict

# 模型路径
unet_model_path = "saved_models/unet/65-gunet.h5"
cnn_model_path = "saved_models/cnn/best_48_from_goodlps.h5"
image_size = (512, 512)
# 加载UNet模型
unet_model = tf.keras.models.load_model(unet_model_path)
# 加载CNN模型
cnn_model = tf.keras.models.load_model(cnn_model_path)


def get_mask(image):
    image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), image_size).astype(np.uint8)
    # 预测
    masks = unet_model.predict(np.array([image]))
    masks = np.squeeze(masks)  # 降维
    mask = np.where(masks > 0.5, 255, 0).astype(np.uint8)  # 二值化，数据类型转换
    return mask


def get_lp_fuc(image, mask):
    image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), image_size).astype(np.uint8)
    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY).astype(np.uint8)
    print("get_lp_fuc->image", image.shape, image.dtype)
    print("get_lp_fuc->mask", mask.shape, mask.dtype)
    cor_lps, cor_lp_masks = get_lp(np.array([image]), np.array([mask]))
    cor_lp = np.squeeze(cor_lps)  # 降维
    return cor_lp


def predict(cor_lp):
    cor_lp = cv2.resize(cor_lp, (128, 48), interpolation=cv2.INTER_AREA)  # 128 48
    print("predict->cor_lp", cor_lp.shape, cor_lp.dtype)
    cor_lp = np.array([cor_lp], dtype=np.uint8)

    pre, probability = cnn_predict(cnn_model, cor_lp)
    acc_num = np.sum(probability > 0.9)
    if acc_num < 5:
        string = "未识别车牌！！！"
        pre = "pre:未识别车牌"
        acc = "acc:0"
    else:
        string = f"pre:{pre[0]}acc:{list(probability)[0]}"
        pre = f"pre:{pre[0]}"
        acc = f"acc:{list(probability)[0]}"
    return pre, acc


# 0.95687395
# 255
if __name__ == '__main__':
    with gr.Blocks() as demo:
        gr.HTML(
            """
            <h1 align="center">LPR</h1>
            <p align="center">这是一个简单的车牌识别系统，加载图片试试效果吧</p>
            <img src="https://huggingface.co/spaces/course-demos/Rick_and_Morty_QA/resolve/main/rick.png " width=250px style="margin: 0 auto;">""")
        with gr.Row():
            image_input = gr.Image()
            image_output = gr.Image()
        image_button = gr.Button("获取掩码")

        with gr.Row():
            image_output1 = gr.Image()
            with gr.Column():
                label = gr.Label("label")
                acc = gr.Label("acc")
        with gr.Row():
            image_button1 = gr.Button("获取车牌")
            image_button2 = gr.Button("识别车牌")

        image_button.click(get_mask, inputs=image_input, outputs=image_output)
        image_button1.click(get_lp_fuc, inputs=[image_input, image_output], outputs=image_output1)
        image_button2.click(predict, inputs=image_output1, outputs=[label, acc])
    demo.launch()
