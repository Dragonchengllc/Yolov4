
# VOC2007下面生成  annotations   imagesets   jpegimage文件夹
# 将 json文件转成xml文件
# 需要修改4处

import os
import numpy as np
import codecs
import json
from glob import glob
import cv2
import shutil
from sklearn.model_selection import train_test_split


# 1.标签路径
labels_source_path = "E:/projects/yolov4/darknet-master/build/darknet/x64/data/voc/VOCdevkit/voc2007/labels/"
labels_path = "E:/projects/yolov4/VOC1/labels/"                      # bai yolov4数据集 labels存储
image_train_path = "E:/projects/yolov4/VOC1/Images/train/"
image_val_path = "E:/projects/yolov4/VOC1/Images/val/"
# 2.创建要求文件夹
# yolo v4
if not os.path.exists(labels_path + "train"):
    os.makedirs(labels_path + "train")
if not os.path.exists(labels_path + "val"):
    os.makedirs(labels_path + "val")

# 获取train的文件名
total_files = glob(image_train_path + "*.bmp")
total_files = [i.replace("\\", "/").split("/")[-1] for i in total_files]
train_txt_files = [i.split(".")[0] for i in total_files]
print("train_txt_files",train_txt_files)
print("length of train_txt_files",len(train_txt_files))
# train
# YOLOV4   train文件复制                                                                                  # ......bai
for train_txt in train_txt_files:
    train_txt = "{}.txt".format(train_txt)
    shutil.copyfile(labels_source_path + train_txt, labels_path + "train/" + train_txt)

# 获取的文件名val
total_files = glob(image_val_path + "*.bmp")
total_files = [i.replace("\\", "/").split("/")[-1] for i in total_files]
val_txt_files = [i.split(".")[0] for i in total_files]
print("val_text_files",val_txt_files)
print("length of val_txt_files",len(val_txt_files))
# YOLOV4         val文件复制                                                                                  # ......bai
for val_txt in val_txt_files:
    val_txt = "{}.txt".format(val_txt)
    shutil.copyfile(labels_source_path + val_txt, labels_path + "val/" + val_txt)

