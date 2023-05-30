import cv2

img = cv2.imread(
    r"D:\Desktop\license plate recognition\My_LPR\data\labelme\train_labels\01-86_91-298&341_449&414-458&394_308&410_304&357_454&341-0_0_14_28_24_26_29-124-24_json.png")

print(img.shape)