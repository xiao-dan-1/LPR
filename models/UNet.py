from tensorflow.keras import layers, models
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization, Dropout

Input_shape = (512, 512, 3)


def conv_33_ReLU(x, filters, kernel_size=(3, 3), strides=(1, 1), padding="same"):
    x = Conv2D(filters, kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.2)(x)
    return x


def copy_crop(old_conv, up_conv):
    concat = layers.concatenate([old_conv, up_conv], axis=3)
    return concat


def max_pool_22(x):
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(x)
    return x


def pu_conv(x, filters, kernel_size=(3, 3), strides=(2, 2), padding="same"):
    x = layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(x)
    x = layers.BatchNormalization(axis=3)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.2)(x)
    return x


def UNet():
    in_put = layers.Input(shape=Input_shape)
    conv1 = conv_33_ReLU(in_put, 8)
    conv1 = conv_33_ReLU(conv1, 8)
    pool1 = max_pool_22(conv1)

    conv2 = conv_33_ReLU(pool1, 16)
    conv2 = conv_33_ReLU(conv2, 16)
    pool2 = max_pool_22(conv2)

    conv3 = conv_33_ReLU(pool2, 32)
    conv3 = conv_33_ReLU(conv3, 32)
    pool3 = max_pool_22(conv3)

    conv4 = conv_33_ReLU(pool3, 64)
    conv4 = conv_33_ReLU(conv4, 64)
    pool4 = max_pool_22(conv4)

    conv5 = conv_33_ReLU(pool4, 128)
    conv5 = conv_33_ReLU(conv5, 128)
    pu_conv1 = pu_conv(conv5, 64)

    concat1 = copy_crop(conv4, pu_conv1)
    conv6 = conv_33_ReLU(concat1, 64)
    conv6 = conv_33_ReLU(conv6, 64)
    pu_conv2 = pu_conv(conv6, 32)

    concat2 = copy_crop(conv3, pu_conv2)
    conv7 = conv_33_ReLU(concat2, 32)
    conv7 = conv_33_ReLU(conv7, 32)
    pu_conv3 = pu_conv(conv7, 16)

    concat3 = copy_crop(conv2, pu_conv3)
    conv8 = conv_33_ReLU(concat3, 16)
    conv8 = conv_33_ReLU(conv8, 16)
    pu_conv4 = pu_conv(conv8, 8)

    concat4 = copy_crop(conv1, pu_conv4)
    conv9 = conv_33_ReLU(concat4, 8)
    conv9 = conv_33_ReLU(conv9, 8)

    out_put = layers.Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv9)

    model = models.Model(in_put, out_put)

    model.summary()

    return model
