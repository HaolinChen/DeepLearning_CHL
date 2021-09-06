# 2021/09/06
# This code can automatically add original pictures by searching the num of pictures in val(train).txt
# HaolinChen_hit@qq.com
import os
import shutil

add_root = './original'
search_root = './pc'

if not os.path.exists(add_root):
    os.makedirs(add_root)

if __name__ == '__main__':
    txt_path = 'F:/DeepLearning/data_set/VOCdevkit/aideck_person1/ImageSets/Main/val.txt'
    image_path = 'F:/DeepLearning/data_set/VOCdevkit/aideck_person1/JPEGImages'
    image_name = '/%s.jpg'
    ids = list()
    l = os.listdir(search_root)

    line_nums = [x.split('.')[0] for x in l]

    with open(txt_path, 'r') as f:
        imgIds = [x for x in f.read().splitlines()]

    for line in line_nums:
        shutil.copy(image_path + image_name % imgIds[int(line)], add_root + image_name % line)
