import os.path
import cv2
import numpy as np
from matplotlib import pyplot as plt
from pycocotools.coco import COCO

'''
COCO结构：
    "images"        : [{"id": 0,"width": 720,"height": 1160,"file_name":},{},]
    "categories"    : [{"id": 0,"name": "car"},{}],
    "annotations"   : [{"id": 0,"image_id": 0,"category_id": 0,"segmentation": [],"bbox": []},{}],
    "info"          : {}
'''
dataset_path = '../datasets/label-studio'
# 构建coco对象
coco = COCO(os.path.join(dataset_path, "result.json"))  # 导入数据集
# info
print('info'.center(50, '-'))
print(coco.info())

# 获取图像id
print('获取图像id'.center(50, '-'))
img_id = coco.getImgIds()
imgs_keys = coco.imgs.keys()
img_kets2 = coco.imgToAnns.keys()  # （用来过滤没有标签的样本）
print(img_id, len(imgs_keys))
print(coco.imgToAnns.keys(), len(coco.imgToAnns.keys()))

# 加载图片
print('加载图片'.center(50, '-'))
img = coco.loadImgs(img_id)
print(img)

# 获取类别id
print('获取类别id'.center(50, '-'))
cat_id = coco.getCatIds()
print(f"cat_id      :{cat_id}   {len(cat_id)}")

# 加载类型内容
print('加载类型内容'.center(50, '-'))
cats = coco.loadCats(cat_id)
print(f"cats        :{cats} {len(cats)}")

# 获取注解id
print('获取注解id'.center(50, '-'))
ann_id = coco.getAnnIds()
print(f"ann_id      :{ann_id}  {len(ann_id)}")

# 加载注解内容
print('加载注解'.center(50, '-'))
ann = coco.loadAnns(ann_id)
print(f"ann      :{ann}")

# 展示注解
print('展示注解'.center(50, '-'))
img_id = coco.getImgIds()
imgs = coco.loadImgs(img_id)

for img in imgs:
    img_path = os.path.join(dataset_path, img.get('file_name'))
    img_id = img.get('id')
    img = plt.imread(img_path)
    plt.imshow(img)
    ann = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
    coco.showAnns(ann)
    plt.show()
    break

# 注解转掩码
print('注解转掩码'.center(50, '-'))
img_ids = coco.getImgIds()
ann_ids = coco.getAnnIds(imgIds=img_ids)
anns = coco.loadAnns(ann_ids)
print(anns)

for ann in anns:
    image_id = ann.get("image_id")
    file_name = coco.loadImgs(image_id)[0].get("file_name")
    mask = coco.annToMask(ann)
    img = plt.imread(os.path.join(dataset_path, file_name))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    plt.imshow(np.concatenate((img, mask * 255), axis=1))
    plt.show()
    cv2.imwrite(f"label_{os.path.split(file_name)[-1]}", mask * 255)
    break
