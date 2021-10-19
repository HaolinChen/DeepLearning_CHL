import os
import shutil
import random
import cv2
import numpy as np
from lxml.etree import Element, SubElement, tostring
from os.path import join
from xml.dom.minidom import parseString


#划分训练集和测试集
src = 'total_qy'
sub_folder = 'images'
sub_folder_label = 'labels_gesture'
dst = 'split_qy'
p_val = 0.2
try:
    shutil.rmtree(dst)
except:
    pass
os.mkdir(dst)
os.mkdir(dst + os.sep + 'train')
os.mkdir(dst + os.sep + 'train' + os.sep + 'ImageSets')
os.mkdir(dst + os.sep + 'train' + os.sep + 'ImageSets' + os.sep + 'Main')
os.mkdir(dst + os.sep + 'train' + os.sep + 'Annotations')
os.mkdir(dst + os.sep + 'train' + os.sep + 'JPEGImages')
os.mkdir(dst + os.sep + 'val')
os.mkdir(dst + os.sep + 'val' + os.sep + 'ImageSets')
os.mkdir(dst + os.sep + 'val' + os.sep + 'ImageSets' + os.sep + 'Main')
os.mkdir(dst + os.sep + 'val' + os.sep + 'Annotations')
os.mkdir(dst + os.sep + 'val' + os.sep + 'JPEGImages')

## converts the normalized positions  into integer positions
def unconvert(class_id, width, height, x, y, w, h):
    xmax = int((x * width) + (w * width) / 2.0)
    xmin = int((x * width) - (w * width) / 2.0)
    ymax = int((y * height) + (h * height) / 2.0)
    ymin = int((y * height) - (h * height) / 2.0)
    class_id = int(class_id)
    return (class_id, xmin, xmax, ymin, ymax)

def xml_transform(id,in_img, in_label, out_img, out_label, dataset_type):

    with open(dst + os.sep + 'val' + os.sep + 'ImageSets' + os.sep + 'Main'+ os.sep + dataset_type + '.txt', "w") as txt_f:

            img = cv2.imread(in_img)
            height, width, channels = img.shape  # pega tamanhos e canais das images

            node_root = Element('annotation')
            node_folder = SubElement(node_root, 'folder')
            node_folder.text = 'JPEGImages'

            node_filename = SubElement(node_root, 'filename')
            node_filename.text = in_img

            node_source = SubElement(node_root, 'source')
            node_database = SubElement(node_source, 'database')
            node_database.text = 'qy database'

            node_size = SubElement(node_root, 'size')
            node_width = SubElement(node_size, 'width')
            node_width.text = str(width)

            node_height = SubElement(node_size, 'height')
            node_height.text = str(height)

            node_depth = SubElement(node_size, 'depth')
            node_depth.text = str(channels)

            node_segmented = SubElement(node_root, 'segmented')
            node_segmented.text = '0'

            if os.path.exists(in_label):
                label_norm = np.loadtxt(in_label).reshape(-1, 5)

                for i in range(len(label_norm)):
                    labels_conv = label_norm[i]
                    new_label = unconvert(labels_conv[0], width, height, labels_conv[1], labels_conv[2], labels_conv[3],
                                          labels_conv[4])
                    node_object = SubElement(node_root, 'object')
                    node_name = SubElement(node_object, 'name')
                    node_name.text = 'gesture'

                    node_pose = SubElement(node_object, 'pose')
                    node_pose.text = 'Unspecified'

                    node_truncated = SubElement(node_object, 'truncated')
                    node_truncated.text = '0'
                    node_difficult = SubElement(node_object, 'difficult')
                    node_difficult.text = '0'
                    node_bndbox = SubElement(node_object, 'bndbox')
                    node_xmin = SubElement(node_bndbox, 'xmin')
                    node_xmin.text = str(new_label[1])
                    node_ymin = SubElement(node_bndbox, 'ymin')
                    node_ymin.text = str(new_label[3])
                    node_xmax = SubElement(node_bndbox, 'xmax')
                    node_xmax.text = str(new_label[2])
                    node_ymax = SubElement(node_bndbox, 'ymax')
                    node_ymax.text = str(new_label[4])
                    xml = tostring(node_root, pretty_print=True)
                    dom = parseString(xml)

                with open(out_label, "wb") as f:
                    f.write(xml)
                shutil.copy(in_img, out_img)

                print(id, file=txt_f)

if __name__=='__main__':
    for root, dirs, files in os.walk(src + os.sep + sub_folder):
        for cur_file in files:
            full_filename = cur_file.split('.')
            filename = ''.join(full_filename[0:-1])
            ext = full_filename[-1]
            if (ext == 'jpg'):
                tmp = 'val' if(random.random() <= p_val) else 'train'

                src_path = root + os.sep + cur_file
                src_path_label = root.replace(sub_folder,sub_folder_label) + os.sep + filename + '.txt'

                dst_path = dst + os.sep + tmp + os.sep + 'JPEGImages' + os.sep + cur_file
                dst_path_label = dst + os.sep + tmp + os.sep + 'Annotations'+ os.sep + filename + '.xml'

                xml_transform(filename,src_path, src_path_label, dst_path, dst_path_label,tmp)
                print(src_path)

