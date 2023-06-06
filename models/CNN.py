import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model

"""
C:
描述卷积层
tf.keras.layers.Conv2D(
filters=卷积核个数,
kernel_size=卷积核尺寸,#正方形写核长整数，或（核高h，核宽w）
strides=滑动步长, t#横纵向相同写步长整数，或(纵向步长h，横向步长w)，慧
padding =“same”or "valid""#使用全零填充是"same"，不使用是“valid”（默
activation = " relu " or " sigmoid " or " tanh " or " softmax"等,#如有BN此处不写
input_shape =(高,宽,通道数)#输入特征图维度，可省略
"""
characters = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫",
              "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "0", "1", "2",
              "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M",
              "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]


class CNN_class():
    def __init__(self):
        self.c1 = layers.Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding="same",
                                input_shape=(70, 220, 3), activation="relu")  # C (核：16*3*3，步长：1，填充：same)
        self.p1 = layers.MaxPool2D(pool_size=(2, 2), strides=2, padding="same")  # P (max,核：2*2，步长：2，填充：same)
        self.d1 = layers.Dropout(0.2)  # D (None)

        self.c21 = layers.Conv2D(filters=32 * 1, kernel_size=(3, 3), strides=1, padding="valid",
                                 activation="relu")  # C (核：32*3*3，步长：1，填充：valid)
        self.c22 = layers.Conv2D(filters=32 * 1, kernel_size=(3, 3), strides=1, padding="valid",
                                 activation="relu")  # C (核：32*3*3，步长：1，填充：valid)
        self.p2 = layers.MaxPool2D(pool_size=(2, 2), strides=2, padding="same")  # P (max,核：2*2，步长：2，填充：same)
        self.d2 = layers.Dropout(0.2)  # D (None)

        self.c31 = layers.Conv2D(filters=32 * 2, kernel_size=(3, 3), strides=1, padding="valid",
                                 activation="relu")  # C (核：32*3*3，步长：1，填充：valid)
        self.c32 = layers.Conv2D(filters=32 * 2, kernel_size=(3, 3), strides=1, padding="valid",
                                 activation="relu")  # C (核：32*3*3，步长：1，填充：valid)
        self.p3 = layers.MaxPool2D(pool_size=(2, 2), strides=2, padding="same")  # P (max,核：2*2，步长：2，填充：same)
        self.d3 = layers.Dropout(0.2)  # D (None)

        self.c41 = layers.Conv2D(filters=32 * 4, kernel_size=(3, 3), strides=1, padding="valid",
                                 activation="relu")  # C (核：32*3*3，步长：1，填充：valid)
        self.c42 = layers.Conv2D(filters=32 * 4, kernel_size=(3, 3), strides=1, padding="valid",
                                 activation="relu")  # C (核：32*3*3，步长：1，填充：valid)
        self.p4 = layers.MaxPool2D(pool_size=(2, 2), strides=2, padding="same")  # P (max,核：2*2，步长：2，填充：same)
        self.d4 = layers.Dropout(0.2)  # D (None)

        self.f = layers.Flatten()

        self.out1 = layers.Dense(units=65, activation="softmax")
        self.out2 = layers.Dense(units=65, activation="softmax")
        self.out3 = layers.Dense(units=65, activation="softmax")
        self.out4 = layers.Dense(units=65, activation="softmax")
        self.out5 = layers.Dense(units=65, activation="softmax")
        self.out6 = layers.Dense(units=65, activation="softmax")
        self.out7 = layers.Dense(units=65, activation="softmax")


def CNN():
    model = CNN_class()
    Input = layers.Input((70, 220, 3))  # 车牌图片shape(80,240,3)
    x = model.c1(Input)
    x = model.p1(x)
    x = model.d1(x)

    x = model.c21(x)
    x = model.c22(x)
    x = model.p2(x)
    x = model.d2(x)

    x = model.c31(x)
    x = model.c32(x)
    x = model.p3(x)
    x = model.d3(x)

    x = model.c41(x)
    x = model.c42(x)
    x = model.p4(x)
    x = model.d4(x)

    x = model.f(x)
    Output = [model.out1(x), model.out2(x), model.out3(x), model.out4(x), model.out5(x), model.out6(x), model.out7(x)]

    model = Model(inputs=Input, outputs=Output)
    return model


if __name__ == '__main__':
    pass
