
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
labelme_path = "E:/projects/yolov4/data/data123/"                      # bai 原始labelme标注数据路径
saved_path = "E:/projects/yolov4/data/data123/VOC2007/"                # bai 保存路径

# 2.创建要求文件夹

# yolo v3
if not os.path.exists(saved_path + "Annotations"):
    os.makedirs(saved_path + "Annotations")
if not os.path.exists(saved_path + "JPEGImages/"):
    os.makedirs(saved_path + "JPEGImages/")
if not os.path.exists(saved_path + "ImageSets/Main/"):
    os.makedirs(saved_path + "ImageSets/Main/")
# yolo v4

if not os.path.exists(saved_path + "JPEGImages/train"):
    os.makedirs(saved_path + "JPEGImages/train")
if not os.path.exists(saved_path + "JPEGImages/val"):
    os.makedirs(saved_path + "JPEGImages/val")

# 3.获取待处理文件
files = glob(labelme_path + "*.json")
print("@@@files...", files)
# files = [i.split("/")[-1].split(".json")[0] for i in files]
files = [i.replace("\\","/").split("/")[-1].split(".json")[0] for i in files]
print("@@@split123...", files)
# 4.读取标注信息并写入 xml
for json_file_ in files:
    print("@@@json_file_...", json_file_)
    json_filename = labelme_path + json_file_ + ".json"
    json_file = json.load(open(json_filename, "r", encoding="utf-8"))
    height, width, channels = cv2.imread(labelme_path + json_file_ + ".bmp").shape
    # TEAT = os.path.basename(json_file_)
    # print("@@@jTEAT ...", TEAT )
    with codecs.open(saved_path + "Annotations/" + json_file_ + ".xml", "w", "utf-8") as xml:
        print("@@@****** ...", saved_path + "Annotations/" + json_file_ + ".xml")
        xml.write('<annotation>\n')
        xml.write('\t<folder>' + 'UAV_data' + '</folder>\n')
        xml.write('\t<filename>' + json_file_ + ".jpg" + '</filename>\n')
        xml.write('\t<source>\n')
        xml.write('\t\t<database>The UAV autolanding</database>\n')
        xml.write('\t\t<annotation>UAV AutoLanding</annotation>\n')
        xml.write('\t\t<image>flickr</image>\n')
        xml.write('\t\t<flickrid>NULL</flickrid>\n')
        xml.write('\t</source>\n')
        xml.write('\t<owner>\n')
        xml.write('\t\t<flickrid>NULL</flickrid>\n')
        xml.write('\t\t<name>DragonCheng</name>\n')
        xml.write('\t</owner>\n')
        xml.write('\t<size>\n')
        xml.write('\t\t<width>' + str(width) + '</width>\n')
        xml.write('\t\t<height>' + str(height) + '</height>\n')
        xml.write('\t\t<depth>' + str(channels) + '</depth>\n')
        xml.write('\t</size>\n')
        xml.write('\t\t<segmented>0</segmented>\n')
        for multi in json_file["shapes"]:
            points = np.array(multi["points"])
            xmin = min(points[:, 0])
            xmax = max(points[:, 0])
            ymin = min(points[:, 1])
            ymax = max(points[:, 1])
            label = multi["label"]
            if xmax <= xmin:
                pass
            elif ymax <= ymin:
                pass
            else:
                xml.write('\t<object>\n')
                # xml.write('\t\t<name>' + "bubble" + '</name>\n')
                xml.write('\t\t<name>' + label + '</name>\n')                # classname
                xml.write('\t\t<pose>Unspecified</pose>\n')
                xml.write('\t\t<truncated>1</truncated>\n')
                xml.write('\t\t<difficult>0</difficult>\n')
                xml.write('\t\t<bndbox>\n')
                xml.write('\t\t\t<xmin>' + str(xmin) + '</xmin>\n')
                xml.write('\t\t\t<ymin>' + str(ymin) + '</ymin>\n')
                xml.write('\t\t\t<xmax>' + str(xmax) + '</xmax>\n')
                xml.write('\t\t\t<ymax>' + str(ymax) + '</ymax>\n')
                xml.write('\t\t</bndbox>\n')
                xml.write('\t</object>\n')
                print(json_filename, xmin, ymin, xmax, ymax, label)
        xml.write('</annotation>')

# 5.复制图片到 VOC2007/JPEGImages/下
image_files = glob(labelme_path + "*.bmp")                        # bai  图片复制  记得更该文件格式

print("copy image files to VOC007/JPEGImages/")
for image in image_files:
    shutil.copy(image, saved_path + "JPEGImages/")

# 6.split files for txt
txtsavepath = saved_path + "ImageSets/Main/"
ftrainval = open(txtsavepath + '/trainval.txt', 'w')
ftest = open(txtsavepath + '/test.txt', 'w')
ftrain = open(txtsavepath + '/train.txt', 'w')
fval = open(txtsavepath + '/val.txt', 'w')
# total_files = glob("./VOC2007/Annotations/*.xml")
total_files = glob(saved_path + "/Annotations/*.xml")
print("@@@3334...", total_files)
# total_files = [i.split("/")[-1].split(".xml")[0] for i in total_files]
total_files = [i.replace("\\", "/").split("/")[-1].split(".xml")[0] for i in total_files]
print("@@@4444...", total_files)
# test_filepath = ""
for file in total_files:
    ftrainval.write(file + "\n")
# test
# for file in os.listdir(test_filepath):
#    ftest.write(file.split(".jpg")[0] + "\n")
# split
train_files, val_files = train_test_split(total_files, test_size=0.15, random_state=42)           # ......bai
# train
print("@@@train_files...", train_files)
for file in train_files:
    print("@@@train.write...", file)
    ftrain.write(file + "\n")
# YOLOV4   train文件复制                                                                                  # ......bai
for train_image in train_files:
    train_image = "{}.bmp".format(train_image)
    shutil.copyfile(saved_path + "JPEGImages/" + train_image, saved_path + "JPEGImages/train/" + train_image)
# val
for file in val_files:
    print("@@@val.write...", file)
    fval.write(file + "\n")
# YOLOV4         val文件复制                                                                                  # ......bai
for val_image in val_files:
    val_image = "{}.bmp".format(val_image)
    shutil.copyfile(saved_path + "JPEGImages/" + val_image, saved_path + "JPEGImages/val/" + val_image)
ftrainval.close()
ftrain.close()
fval.close()
# ftest.close()
