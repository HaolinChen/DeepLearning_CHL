import os
import numpy as np
import scipy.io as sio
import shutil
from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString
import cv2
from os.path import join

folder_root = 'VOCdevkit/aideck_person2'
yolo_root = 'cust_data_7'
YOLO_CLASSES = ['person']

if not os.path.exists(folder_root):
    os.makedirs(folder_root)


def make_voc_dir():
    # labels 目录若不存在，创建labels目录。若存在，则清空目录
    if not os.path.exists(folder_root + '/Annotations'):
        os.makedirs(folder_root + '/Annotations')
    if not os.path.exists(folder_root + '/ImageSets'):
        os.makedirs(folder_root + '/ImageSets')
        os.makedirs(folder_root + '/ImageSets/Main')
    if not os.path.exists(folder_root + '/JPEGImages'):
        os.makedirs(folder_root + '/JPEGImages')


## converts the normalized positions  into integer positions
def unconvert(class_id, width, height, x, y, w, h):
    xmax = int((x * width) + (w * width) / 2.0)
    xmin = int((x * width) - (w * width) / 2.0)
    ymax = int((y * height) + (h * height) / 2.0)
    ymin = int((y * height) - (h * height) / 2.0)
    class_id = int(class_id)
    return (class_id, xmin, xmax, ymin, ymax)


## converts coco into xml
def xml_transform(root, data_type, classes):
    class_path = join(root, 'labels/' + data_type)
    ids = list()

    l = os.listdir(class_path)

    check = '.DS_Store' in l
    if check == True:
        l.remove('.DS_Store')

    ids = [x.split('.')[0] for x in l]

    annopath = root + '/labels/' + data_type + '/%s.txt'
    imgpath = root + '/images/' + data_type + '/%s.jpg'

    out_path = folder_root + '/Annotations' + '/%s.xml'
    out_imgpath = folder_root + '/JPEGImages' + '/%s.jpg'

    with open(folder_root + '/ImageSets/Main/' + data_type + '.txt', "w") as txt_f:
        for i in range(len(ids)):
            img_id = ids[i]
            print(img_id)
            img = cv2.imread(imgpath % img_id)
            height, width, channels = img.shape  # pega tamanhos e canais das images

            node_root = Element('annotation')
            node_folder = SubElement(node_root, 'folder')
            node_folder.text = 'JPEGImages'
            img_name = img_id + '.jpg'

            node_filename = SubElement(node_root, 'filename')
            node_filename.text = img_name

            node_source = SubElement(node_root, 'source')
            node_database = SubElement(node_source, 'database')
            node_database.text = 'aideck database'

            node_size = SubElement(node_root, 'size')
            node_width = SubElement(node_size, 'width')
            node_width.text = str(width)

            node_height = SubElement(node_size, 'height')
            node_height.text = str(height)

            node_depth = SubElement(node_size, 'depth')
            node_depth.text = str(channels)

            node_segmented = SubElement(node_root, 'segmented')
            node_segmented.text = '0'

            target = (annopath % img_id)
            if os.path.exists(target):
                label_norm = np.loadtxt(target).reshape(-1, 5)

                for i in range(len(label_norm)):
                    labels_conv = label_norm[i]
                    new_label = unconvert(labels_conv[0], width, height, labels_conv[1], labels_conv[2], labels_conv[3],
                                          labels_conv[4])
                    node_object = SubElement(node_root, 'object')
                    node_name = SubElement(node_object, 'name')
                    node_name.text = classes[new_label[0]]

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

                with open(out_path % img_id, "wb") as f:
                    f.write(xml)
                shutil.copy(imgpath % img_id, out_imgpath % img_id)

                print(img_id, file=txt_f)


if __name__ == '__main__':
    make_voc_dir()
    xml_transform(yolo_root, 'train', YOLO_CLASSES)
    xml_transform(yolo_root, 'val', YOLO_CLASSES)
