import os
import cv2
from pycocotools.coco import COCO
import argparse


def coco_to_Mask(coco, out_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    img_ids = coco.getImgIds()
    ann_ids = coco.getAnnIds(imgIds=img_ids)
    anns = coco.loadAnns(ann_ids)
    for ann in anns:
        image_id = ann.get("image_id")
        file_name = coco.loadImgs(image_id)[0].get("file_name")
        mask = coco.annToMask(ann) * 255
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        cv2.imwrite(f"{out_path}/{os.path.split(file_name)[-1]}", mask)
    print("coco_to_Mask Done!!!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str,
                        default="../datasets/label-studio/project-1-at-2023-05-30-17-01-7ce41586",
                        help="data path")
    opt = parser.parse_args()

    save_path = os.path.join(opt.data_path, "labels")

    # 构建coco对象
    coco = COCO(os.path.join(opt.data_path, "result.json"))

    # 掩码生成
    coco_to_Mask(coco, save_path)
