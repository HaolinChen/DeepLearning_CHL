import os

people_name = 'fmn_'
dataset_path = './gesture_fmn'
class_folders = os.listdir(dataset_path)

for num in range(len(class_folders)):
    print(class_folders[num])
    pic_num = 0
    image_folder_path = dataset_path + '/' + class_folders[num]
    IMAGES = os.listdir(image_folder_path)
    for j in range(len(IMAGES)):
        os.rename(image_folder_path + '/' + IMAGES[j],
                  image_folder_path + '/' + people_name + class_folders[num] + '_%d.jpg' % pic_num)
        pic_num = pic_num + 1
        print(pic_num)
