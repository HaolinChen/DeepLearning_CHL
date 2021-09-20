import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2

dataset_name = 'gesture_yz'
dataset_root = './' + dataset_name
cropped_root = './' + dataset_name + '/cropped_'

class_names = ['person', 'gesture']
gesture_names = ['backward', 'down', 'forward', 'left', 'right', 'up']


def get_annotation(image_id):
    annotation_file = dataset_root + f"/Annotations_yz/{image_id}.xml"
    objects = ET.parse(annotation_file).findall("object")
    boxes = []
    labels = []
    is_difficult = []
    for object in objects:
        input_name = object.find('name').text.lower().strip()
        # we're only concerned with clases in our list
        if input_name in class_names:
            bbox = object.find('bndbox')
            # VOC dataset format follows Matlab, in which indexes start from 0
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            boxes.append([x1, y1, x2, y2])
            labels.append(input_name)
            is_difficult_str = object.find('difficult').text
            is_difficult.append(int(is_difficult_str) if is_difficult_str else 0)

    return (np.array(boxes, dtype=np.float32),
            labels,
            np.array(is_difficult, dtype=np.uint8))


if __name__ == '__main__':

    for class_name in class_names:
        if not os.path.exists(cropped_root + class_name):
            os.mkdir(cropped_root + class_name)
        for gesture_name in gesture_names:
            if not os.path.exists(cropped_root + class_name + '/' + gesture_name):
                os.mkdir(cropped_root + class_name + '/' + gesture_name)

    for gesture_name in gesture_names:
        images_path = dataset_root + '/' + gesture_name
        IMAGES = os.listdir(images_path)
        for img in IMAGES:
            img_id = img.split(".", -1)[0]
            img_box, img_label, img_difiicult = get_annotation(img_id)
            orig_img = cv2.imread(images_path + '/' + img)
            for i, label_name in enumerate(img_label):
                cropped_img = orig_img[int(img_box[i][1]):int(img_box[i][3]), int(img_box[i][0]):int(img_box[i][2])]
                path = cropped_root + label_name + '/' + gesture_name + '/' + img
                cv2.imwrite(path, cropped_img)
                print("Img %d cropped done.", img_id)
