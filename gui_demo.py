import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf
from tkmacosx import Button
from utils.LP_Correction import get_lp
from CNN_predict import cnn_predict

# 模型路径
unet_model_path = "saved_models/unet/65-gunet.h5"
cnn_model_path = "saved_models/cnn/best_48_from_goodlps.h5"

# 加载UNet模型
unet_model = tf.keras.models.load_model(unet_model_path)
# 加载CNN模型
cnn_model = tf.keras.models.load_model(cnn_model_path)


def get_mask(image):
    # 预测
    masks = unet_model.predict(np.array([image]))
    masks = np.squeeze(masks)  # 降维
    mask = np.where(masks > 0.5, 255, 0).astype(np.uint8)  # 二值化，数据类型转换
    return mask


def get_lp_fuc(image, mask):
    cor_lps, cor_lp_masks = get_lp(np.array([image]), np.array([mask]))
    cor_lp = np.squeeze(cor_lps)  # 降维
    cor_lp_mask = np.squeeze(cor_lp_masks)  # 降维
    return cor_lp, cor_lp_mask


def predict(cor_lp):
    cor_lp = cv2.resize(cor_lp, (128, 48), interpolation=cv2.INTER_AREA)  # 128 48

    cor_lp = np.array([cor_lp], dtype=np.uint8)

    pre, probability = cnn_predict(cnn_model, cor_lp)
    acc_num = np.sum(probability > 0.9)
    if acc_num < 5:
        string = "未识别车牌！！！"
    else:
        string = f"pre:\n{pre}\nacc:\n{list(probability)[0]}"

    return string


def recognize_lp():
    # 选择图片文件
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])

    if file_path:
        # 读取选择的图片
        image = cv2.imread(file_path)
        image_size = (512, 512)
        image = cv2.resize(image, image_size).astype(np.uint8)

        # 调用get_mask函数进行车牌识别
        mask = get_mask(image)

        # 调用get_lp函数进行车牌提取
        cor_lp_image, cor_lp_mask = get_lp_fuc(image, mask)

        # 显示加载的图片
        input_img = Image.open(file_path)
        input_img = input_img.resize((300, 200))
        input_img_tk = ImageTk.PhotoImage(input_img)
        input_label.configure(image=input_img_tk)
        input_label.image = input_img_tk

        # 显示识别结果的图像
        result_img = Image.fromarray(mask)
        result_img = result_img.resize((300, 200))
        result_img_tk = ImageTk.PhotoImage(result_img)
        result_label.configure(image=result_img_tk)
        result_label.image = result_img_tk

        # 显示提取的车牌图像
        cor_lp_img = Image.fromarray(cor_lp_image)
        cor_lp_img = cor_lp_img.resize((128, 48))
        cor_lp_img_tk = ImageTk.PhotoImage(cor_lp_img)
        cor_lp_label.configure(image=cor_lp_img_tk)
        cor_lp_label.image = cor_lp_img_tk

        # 调用预测函数进行车牌预测
        prediction = predict(cor_lp_image)
        # 清空文本框并重新启用
        prediction_text.config(state="normal")
        prediction_text.delete("1.0", tk.END)
        prediction_text.insert(tk.END, prediction)
        # 禁用文本框，防止用户编辑内容
        prediction_text.config(state="disabled")
        print(prediction)


def clear():
    input_label.configure(image=None)
    input_label.image = None
    result_label.configure(image=None)
    result_label.image = None
    cor_lp_label.configure(image=None)
    cor_lp_label.image = None
    # 清空文本框并重新启用
    prediction_text.config(state="normal")
    prediction_text.delete("1.0", tk.END)
    # 禁用文本框，防止用户编辑内容
    prediction_text.config(state="disabled")


def quit_app():
    window.quit()


# 创建主窗口
window = tk.Tk()
window.title("车牌识别")
window.geometry("800x600")
window.config(bg="#F5F5F5")

# 创建顶部Frame容器
top_frame = tk.Frame(window, bg="#F5F5F5")
top_frame.pack(pady=20)

# 创建选择图片按钮
select_button = Button(top_frame, text="选择图片", font=("Helvetica", 16), bg="#007AFF", fg="white", padx=20, pady=10,
                       command=recognize_lp)
select_button.pack(side="left", padx=20)

# 创建清除按钮
clear_button = Button(top_frame, text="清除", font=("Helvetica", 16), bg="#FF3B30", fg="white", padx=20, pady=10,
                      command=clear)
clear_button.pack(side="left", padx=20)

# 创建退出按钮
quit_button = Button(top_frame, text="退出", font=("Helvetica", 16), bg="#FF3B30", fg="white", padx=20, pady=10,
                     command=quit_app)
quit_button.pack(side="left", padx=20)

# 创建中间Frame容器
middle_frame = tk.Frame(window, bg="#F5F5F5")
middle_frame.pack()

# 创建左侧显示区域
left_frame = tk.Frame(middle_frame, bg="#F5F5F5")
left_frame.pack(side="left", padx=20)

# 创建右侧显示区域
right_frame = tk.Frame(middle_frame, bg="#F5F5F5")
right_frame.pack(side="right", padx=20)

# 创建加载图片的标题标签
input_title_label = tk.Label(left_frame, text="原始图像", font=("Helvetica", 14), bg="#F5F5F5")
input_title_label.pack()

# 创建加载图片的标签
input_label = tk.Label(left_frame, bg="white", bd=2, relief="solid", width=300, height=200)
input_label.pack(pady=10)

# 创建识别结果的标题标签
result_title_label = tk.Label(left_frame, text="掩码图像", font=("Helvetica", 14), bg="#F5F5F5")
result_title_label.pack()

# 创建识别结果显示的标签
result_label = tk.Label(left_frame, bg="white", bd=2, relief="solid", width=300, height=200)
result_label.pack(pady=10)

# 创建提取车牌的标题标签
cor_lp_title_label = tk.Label(right_frame, text="提取车牌", font=("Helvetica", 14), bg="#F5F5F5")
cor_lp_title_label.pack()

# 创建提取车牌显示的标签
cor_lp_label = tk.Label(right_frame, bg="white", bd=2, relief="solid", width=300, height=48)
cor_lp_label.pack(pady=10)

# 创建车牌预测的标题标签
prediction_title_label = tk.Label(right_frame, text="预测结果", font=("Helvetica", 14), bg="#F5F5F5")
prediction_title_label.pack()

# 创建车牌预测显示的文本框
prediction_text = tk.Text(right_frame, font=("Helvetica", 14), bg="white", bd=2, relief="solid", width=40, height=6)
prediction_text.pack(pady=10)

# 设置文本框为只读
prediction_text.config(state="disabled")

# 运行窗口主循环
window.mainloop()
