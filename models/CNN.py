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


def CNN():
    def conv_block(inputs, filters):
        conv1 = layers.Conv2D(filters=filters, kernel_size=3, padding='same')(inputs)  # 卷积层1
        conv2 = layers.Conv2D(filters=filters, kernel_size=3, padding='same')(conv1)  # 卷积层1
        bn1 = layers.BatchNormalization()(conv1)  # BN层1
        relu1 = layers.Activation('relu')(bn1)  # 激活层1
        pool1 = layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(relu1)
        drop1 = layers.Dropout(0.3)(pool1)  # dropout层
        return drop1

    # 输入层
    filter = 32
    Input = layers.Input((48, 128, 3))
    x = conv_block(Input, filter)
    x = conv_block(x, filter * 2)
    x = conv_block(x, filter * 2 * 2)
    x = conv_block(x, filter * 2 * 2 * 2)
    x = conv_block(x, filter * 2 * 2 * 2 * 2)
    # 全连接层
    x = layers.Flatten()(x)
    # 输出层
    out1 = layers.Dense(units=65, activation="softmax")(x)
    out2 = layers.Dense(units=65, activation="softmax")(x)
    out3 = layers.Dense(units=65, activation="softmax")(x)
    out4 = layers.Dense(units=65, activation="softmax")(x)
    out5 = layers.Dense(units=65, activation="softmax")(x)
    out6 = layers.Dense(units=65, activation="softmax")(x)
    out7 = layers.Dense(units=65, activation="softmax")(x)

    Output = [out1, out2, out3, out4, out5, out6, out7]

    model = Model(inputs=Input, outputs=Output)

    return model


if __name__ == '__main__':
    pass
