import os, sys

import matplotlib.pyplot as plt

dir_mytest = "/"
sys.path.insert(0, dir_mytest)
from utils.LoadDatasets import LoadData
import tensorflow as tf
from utils.Progressbar import Progressbar


# 将根据数据的具体格式进行编译，并存储为tfrecords格式文件。
# 其中float格式编译为float_list，字符串、高维array等格式直接编译为bytes_list格式。
def save_tfrecords(data, label, desfile):
    with tf.io.TFRecordWriter(desfile) as writer:
        for i in range(len(data)):
            features = tf.train.Features(
                feature={
                    "image": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(data[i]).numpy()])),
                    "label": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(label[i]).numpy()])),
                }
            )
            example = tf.train.Example(features=features)
            serialized = example.SerializeToString()
            writer.write(serialized)


def tfrecords_decode(example):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string),
    }
    parsed_example = tf.io.parse_single_example(example, features=feature_description)

    x_sample = tf.io.parse_tensor(parsed_example['image'], tf.uint8)
    y_sample = tf.io.parse_tensor(parsed_example['label'], tf.uint8)

    return x_sample, y_sample


def load_dataset(filepaths, batch_size=16):
    shuffle_buffer_size = 128

    dataset = tf.data.TFRecordDataset(filepaths)
    dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.map(map_func=tfrecords_decode, num_parallel_calls=8)
    dataset = dataset.batch(batch_size).prefetch(1)

    return dataset


# train_set = load_dataset(["path1.tfrecords", "path2.tfrecords"])
#
# valid_set = load_dataset(["path3.tfrecords", "path4.tfrecords"])


if __name__ == '__main__':
    # 基本参数
    data_path = "../../../CCPD/my_use"
    image_folder = 'Images'
    mask_folder = 'Mask'
    images_path = os.path.join(data_path, image_folder)
    annotations_path = os.path.join(data_path, mask_folder)
    input_shape = (512, 512, 3)
    image_size = (512, 512)
    validation_split = 0.2
    batch_size = 4

    # # 加载数据集
    # X_train, X_test, y_train, y_test = LoadData(images_path, annotations_path, image_size,
    #                                             validation_split=validation_split, num=5000)
    #
    # save_tfrecords(X_train, y_train, "train.tfrecords")
    # save_tfrecords(X_test, y_test, "test.tfrecords")

    train_sets_batch = load_dataset("train.tfrecords")

    for train_sets in Progressbar(train_sets_batch):
        images, labels = train_sets
        # print(images.shape, images.dtype)
        # print(labels.shape, labels.dtype)
        for image, label in zip(images, labels):
            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.subplot(1, 2, 2)
            plt.imshow(label)
            plt.show()
        break
