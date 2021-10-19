import os
import shutil

# 把各个子数据集的数据拷贝至total

src = 'dataset0009'
dst = 'total_qy'

dataset_number = os.path.realpath(src).split(os.sep)[-1][-4:]

for root, dirs, files in os.walk(src):
    for cur_file in files:
        full_filename = cur_file.split('.')
        filename = ''.join(full_filename[0:-1])
        ext = full_filename[-1]
        if (ext == 'txt' or ext == 'jpg') and len(filename) == 5:  # 只选择符合规定扩展名，同时文件名位数为5的文件
            full_dir = root.split(os.sep)
            cur_dir = full_dir[-1]
            src_path = root + os.sep + cur_file
            dst_path = dst + os.sep + cur_dir + os.sep + 's' + dataset_number + 'p' + filename + '.' + ext
            shutil.copy(src_path, dst_path)
