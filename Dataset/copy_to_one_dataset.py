import os
import shutil

#可以把多个目录下的图像按顺序放到同一个目录下

src = 'D://tmp'
dst = 'dataset0002/images'
dataset_number = os.path.realpath(src).split(os.sep)[-1][-4:]

cnt = 0
for root, dirs, files in os.walk(src): 
    for cur_file in files:
        full_filename = cur_file.split('.')
        filename = ''.join(full_filename[0:-1])
        ext = full_filename[-1]
        if (ext == 'jpg') and len(filename) == 5: #只选择符合规定扩展名，同时文件名位数为5的文件
            full_dir = root.split(os.sep)
            cur_dir = full_dir[-1]
            src_path = root + os.sep + cur_file
            name = '%05d'%cnt+'.jpg'
            dst_path = dst + os.sep + name
            shutil.copy(src_path,dst_path)
            cnt += 1
            print(cnt)