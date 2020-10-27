import os
import sys
import glob2
import cv2

import xml.etree.ElementTree as ET

from tqdm import tqdm 

def load_class_names(filename):
    with open(filename, 'r', encoding='utf8') as f:
        classes = [line.strip() for line in f.readlines()]
    return classes

# convert xml to json 
def get_data_dicts_xml(root_dir, dataset_dir, label_dir, img_dir, BoxMode):
    anno_files = glob2.glob(os.path.join(root_dir, dataset_dir, label_dir, "*.xml"))
    path_classes_file = os.path.join(root_dir, dataset_dir, "labels.txt")
    classes = load_class_names(path_classes_file)

    dataset_dicts = []
    count = 0
    len_anno = len(anno_files)
    with tqdm(total=len_anno) as pbar:
      for file_path in anno_files:
          record = {}
          try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            filename = root.findtext('filename')

            img_path = os.path.join(root_dir, dataset_dir, img_dir, filename)

            height, width = cv2.imread(img_path).shape[:2]
            
            record["file_name"] = img_path
            record["image_id"] = count
            record["height"] = height
            record["width"] = width

            count += 1

            objs = []
            for obj in root.iter('object'):
              name = obj.findtext('name')
              bbox = obj.find('bndbox')
              xmin = int(bbox.findtext('xmin'))
              xmax = int(bbox.findtext('xmax'))
              ymin = int(bbox.findtext('ymin'))
              ymax = int(bbox.findtext('ymax'))

              obj = {
                    'bbox': [xmin, ymin, xmax, ymax],
                    'bbox_mode': BoxMode.XYXY_ABS,
                    'category_id': classes.index(name),
                    "iscrowd": 0
              }
              objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
            pbar.update(1)
          except Exception as e:
            print(e)
            pass

    return dataset_dicts

# convert txt to json
def get_data_dicts_txt(root_dir, dataset_dir, label_dir, img_dir, BoxMode):
    anno_files = glob2.glob(os.path.join(root_dir, dataset_dir, label_dir, "*.txt"))
    path_classes_file = os.path.join(root_dir, dataset_dir, "labels.txt")
    classes = load_class_names(path_classes_file)

    dataset_dicts = []
    count = 0
    len_anno = len(anno_files)
    with tqdm(total=len_anno) as pbar:
      for file_path in anno_files:
          record = {}      
          try:
            filename = file_path.split("/")[-1]
            filename = filename.split(".")[0]
            img_path = os.path.join(root_dir, dataset_dir, img_dir, filename + '.jpg')
            height, width = cv2.imread(img_path).shape[:2]
            
            record["file_name"] = img_path
            record["image_id"] = count
            record["height"] = height
            record["width"] = width

            count += 1

            annotations = open(file_path, 'r')
            objs = []
            for line in annotations:
              line = line.rstrip('\n')
              class_id, x_center, y_center, w, h = line.split()[:]
              w = int(float(w) * width)
              h = int(float(h) * height)
              xmin = int((float(x_center) * width) - w/2)
              ymin = int((float(y_center) * height) - h/2)
              xmax = xmin + w
              ymax = ymin + h

              obj = {
                    'bbox': [xmin, ymin, xmax, ymax],
                    'bbox_mode': BoxMode.XYXY_ABS,
                    'category_id': int(class_id),
                    "iscrowd": 0
              }

              objs.append(obj)

            record["annotations"] = objs
            dataset_dicts.append(record)
            pbar.update(1)
          except Exception as e:
            print(e)
            pass

    return dataset_dicts