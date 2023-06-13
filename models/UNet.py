import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Conv2DTranspose, \
    concatenate


def Unet(input_shape, depth=4, filters=16, dtype=tf.float16):
    # 输入层
    inputs = Input(input_shape)

    def conv_33_relu(Input, filters):
        conv1 = Conv2D(filters, 3, padding='same')(Input)
        bn1 = BatchNormalization()(conv1)
        Output = Activation('relu', dtype=dtype)(bn1)
        return Output

    def conv_block(inputs, filters):
        conv1 = conv_33_relu(inputs, filters)
        conv2 = conv_33_relu(conv1, filters)
        pool = MaxPooling2D(pool_size=(2, 2))(conv2)
        return conv2, pool

    def upconv_block(inputs, skip_features, filters):
        upsample = Conv2DTranspose(filters, 2, strides=(2, 2), padding='same')(inputs)
        concat = concatenate([skip_features, upsample], axis=3)

        conv1 = conv_33_relu(concat, filters)
        conv2 = conv_33_relu(conv1, filters)
        return conv2

    skip_features = []
    x = inputs
    # 编码器部分
    for _ in range(depth):
        conv, x = conv_block(x, filters)
        skip_features.append(conv)
        filters *= 2

    x = conv_33_relu(x, filters)
    x = conv_33_relu(x, filters)

    # 解码器部分
    filters //= 2  # //取整数
    for i in range(depth - 1, -1, -1):
        x = upconv_block(x, skip_features[i], filters)
        filters //= 2

    # 输出层
    outputs = Conv2D(1, 1, activation='sigmoid')(x)

    # 创建模型
    model = Model(inputs=inputs, outputs=outputs)

    return model
