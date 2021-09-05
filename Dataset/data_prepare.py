# coding: utf-8

import numpy as np
import cv2
import random
import os
# import albumentations as albu


"""
    随机挑选CNum张图片，进行按通道计算均值mean和标准差std
    先将像素从0～255归一化至 0-1 再计算
"""


# train_txt_path = os.path.join("F:/RGBD Study/1 augmentation/pre-depth/dataset/RGBD_for_train/train_rgbd.txt")
# depth_path = 'F:/RGBD Study/1 augmentation/pre-depth/equalHistdepth/'

CNum = 2025     # 挑选多少图片进行计算

img_h, img_w = 352, 352
imgs = np.zeros([img_w, img_h, 3, 1])
means, stdevs = [], []

# read a txt
# with open(train_txt_path, 'r') as f:
#     depthlist = []
#     lines = [i.strip() for i in f.readlines()]
#     for line_num, line in enumerate(lines):
#         list1 = line.split("../",3)[2]
#         list1 = list1.split("/",5)[4]
#         depthlist.append(list1)

    # random.shuffle(lines)   # shuffle , 随机挑选图片F:\RGBD Study\1 augmentation\pre-depth\dataset\Depth
dataset_path = 'F:/RGBD Study/1 augmentation/pre-depth/dataset/'
    # testdatasets = ['RGB', 'GT', 'depth']
testdatasets = ['Depth']

    # lst = '../list/'
    # compute = 'rgbImg_global_contrast'
    # compute = '0722'

for dataset in testdatasets:
        # save_path = compute + dataset + '/'
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)

        # lst = dataset_path + dataset + '/depth/'

    lst = dataset_path + dataset + '/'
    datasets_lists = os.listdir(lst)

    for dataset_list in datasets_lists:
        # print('starting analyze {}'.format(dataset_list))
        # depths = cv2.imread(lst + dataset_list)
        # depths = cv2.resize(depths, (img_h, img_w))

        for i in range(CNum):
        # img_path = depthlist[i].rstrip().split()[0]

            img = cv2.imread(lst + dataset_list)
            img = cv2.resize(img, (img_h, img_w))

            img = img[:, :, :, np.newaxis]
            imgs = np.concatenate((imgs, img), axis=3)
            print(i)

imgs = imgs.astype(np.float32)/255.


for i in range(3):
    pixels = imgs[:,:,i,:].ravel()  # 拉成一行
    means.append(np.mean(pixels))
    stdevs.append(np.std(pixels))

means.reverse() # BGR --> RGB
stdevs.reverse()

print("normMean = {}".format(means))
print("normStd = {}".format(stdevs))
print('transforms.Normalize(normMean = {}, normStd = {})'.format(means, stdevs))