import os
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# rootdir = "C:\\Users\\DeepLearning\\Desktop\\COCO -Text-words-trainval\\train_words"
# mapping_file_path = "C:\\Users\\DeepLearning\\Desktop\\COCO-Text-words-trainval\\train_words_gt.txt"
rootdir = "D:\\COCO-Text-words-trainval\\train_words"
mapping_file_path = "D:\\COCO-Text-words-trainval\\train_words_gt.txt"

file_names = []
for parent, dirnames, filenames in os.walk(rootdir):    #三个参数：分别返回1.父目录 2.所有文件夹名字（不含路径） 3.所有文件名字
    file_names = filenames
    # for filename in filenames:                        #输出文件信息
    #     print("parent is" + parent)
    #     print("filename is:" + filename)
    #     print("the full name of the file is:" + os.path.join(parent, filename))


# read mapping as list
mapping_list = {}
map_file = open(mapping_file_path, encoding='utf-8')

for i in map_file:
    i = i.replace("\n", '')
    a = i.split(',')[0]
    b = i.replace(a+',', "")
    mapping_list[a] = b

def get_train_text_and_image():
    # read a image and get a name than get mapping
    x = random.randint(0, len(file_names) - 1)
    file_name = file_names[x]
    file_path = str(rootdir)
    if not file_path.endswith("\\"):
        file_path += "\\"
    file_path += file_name
    captcha_image = Image.open(file_path)
    captcha_image = captcha_image.resize((256, 64))
    # captcha_image = np.array(captcha_image)

    text, image = mapping_list[file_name.split(".")[0].split("_")[0]], captcha_image
    if len(text) > 16:
        return get_train_text_and_image()
    return text, image

def get_train_text_and_image_with_chou_mi_cai_yang():
    text, image = get_train_text_and_image()
    # box = (0, 0, 16, 16)
    # roi = image.crop(box)
    l = []
    x1 = 0
    x2 = 16
    y1 = 0
    y2 = 16
    n = 16
    while (n<=64):
        while (y2)<=64:
            while (x2)<=256:
                print('x1=' + str(x1) + ',y1=' + str(y1) + ',x2=' + str(x2) + ',y2=' + str(y2))
                box = (x1, y1, x2, y2)
                # if (x2 - x1) == n and (y2 - y1) == n:
                l.append(np.array(image.crop(box)))
                x1 += 4
                x2 += 4
            x1 = 0
            x2 = n
            y1 += 4
            y2 += 4
        n += 8
        x1 = 0
        y1 = 0
        x2 = n
        y2 = n

    for o in l:
        print(o)


    # f = plt.figure()
    # ax = f.add_subplot(111)
    # ax.text(0.1, 0.9, text, ha='center', va='center', transform=ax.transAxes)
    # plt.imshow(image)
    # plt.show()
    # plt.imshow(roi)
    # plt.show()
    pass


get_train_text_and_image_with_chou_mi_cai_yang()