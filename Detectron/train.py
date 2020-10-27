import os
import cv2
import json
import random
import itertools
import numpy as np
import glob2
import copy
import itertools
import torch
import json

import xml.etree.ElementTree as ET

import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.config import get_cfg as config_detectron
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader, build_detection_train_loader
from detectron2.data import detection_utils 
import detectron2.data.transforms as T
from detectron2.structures import BoxMode

from utils.utils import load_class_names, get_data_dicts_txt, get_data_dicts_xml
from utils.parser import get_config
from tqdm import tqdm 

# Create Json
'''
    Input: path images + labels dir
    Output: 
        - save train + val json file
        - return val_dicts, train_dicts
'''
def create_json(root_dir, train_dir, label_train_dir, image_train_dir, \
                val_dir, label_val_dir, image_val_dir, json_val_name, json_train_name, use_xml):
    
    if use_xml:
        val_dicts = get_data_dicts_xml(root_dir, val_dir, label_val_dir, image_val_dir, BoxMode)
        train_dicts = get_data_dicts_xml(root_dir, train_dir, label_train_dir, image_train_dir, BoxMode)
        
    else:
        val_dicts = get_data_dicts_txt(root_dir, val_dir, label_val_dir, image_val_dir, BoxMode)
        train_dicts = get_data_dicts_txt(root_dir, train_dir, label_train_dir, image_train_dir, BoxMode)

    json_val_path = os.path.join(root_dir, val_dir, json_val_name)
    json_train_path = os.path.join(root_dir, train_dir, json_train_name)

    with open(json_val_path, "w") as f:
        json.dump(val_dicts, f)
    
    with open(json_train_path, "w") as f:
        json.dump(train_dicts, f)
    
    return train_dicts, val_dicts

# Read json
'''
    Input: path of train/val json
    Output: train_dicts, val_dicts
'''
def read_json(train_json_path, val_json_path):
    with open(train_json_path, "r")  as f:
        train_dicts = json.load(f)
    
    with open(val_json_path, "r") as f:
        val_dicts = json.load(f)
    
    return train_dicts, val_dicts

def regist_data(root_dir, train_dicts, val_dicts, list_classes):
    for i in range(len(val_dicts)):
        for j in range(len(val_dicts[i]["annotations"])):
            val_dicts[i]["annotations"][j]['bbox_mode'] = BoxMode.XYXY_ABS

    for i in range(len(train_dicts)):
        for j in range(len(train_dicts[i]["annotations"])):
            train_dicts[i]["annotations"][j]['bbox_mode'] = BoxMode.XYXY_ABS
    
    data = [train_dicts, val_dicts]
    dataset_train = os.path.join(root_dir, "train")
    dataset_val = os.path.join(root_dir, "val")

    for index, d in enumerate(["train", "val"]):
        DatasetCatalog.register(os.path.join(root_dir, d), lambda index=index: data[index])
        MetadataCatalog.get(os.path.join(root_dir, d)).set(thing_classes=list_classes)
    
    return dataset_train, dataset_val
    
# Augmentation
def custom_mapper(dataset_dict, size, flip_prob, min_brightness, max_brightness, \
                min_contrast, max_contrast, min_saturation, max_saturation):
    # Implement a mapper, similar to the default DatasetMapper, but with your own customizations
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    image = detection_utils.read_image(dataset_dict["file_name"], format="BGR")
    transform_list = [ 
                    T.Resize(size),
                    T.RandomBrightness(min_brightness, max_brightness),
                    T.RandomContrast(min_contrast, max_contrast),
                    T.RandomSaturation(min_saturation, max_saturation),

                    T.RandomFlip(prob=flip_prob, horizontal=False, vertical=True),
                    T.RandomFlip(prob=flip_prob, horizontal=True, vertical=False), 
                ]
    image, transforms = T.apply_transform_gens(transform_list, image)
    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

    annos = [
        detection_utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop("annotations")
        if obj.get("iscrowd", 0) == 0
    ]
    instances = detection_utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict["instances"] = detection_utils.filter_empty_instances(instances)
    return dataset_dict

# Training with augmentation
class AugmentTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=custom_mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):

        if output_folder is None:
            os.makedirs("Evaluate_dir", exist_ok=True)
            output_folder = "Evaluate_dir"

        return COCOEvaluator(dataset_name, cfg, False, output_folder)

# Training without augmentation
class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):

        if output_folder is None:
            os.makedirs("Evaluate_dir", exist_ok=True)
            output_folder = "Evaluate_dir"

        return COCOEvaluator(dataset_name, cfg, False, output_folder)

def Training(detectron, config_model_path, dataset_train, dataset_val, num_workers, weights_path, \
        image_per_batch, lr, max_iter, batch_size, num_classes, output_dir, eval_period, use_augment):
    
    detectron.merge_from_file(config_model_path)
    detectron.DATASETS.TRAIN  = (dataset_train,)
    detectron.DATASETS.TEST = (dataset_val,)
    detectron.DATALOADER.NUM_WORKERS = num_workers
    detectron.MODEL.WEIGHTS = weights_path
    detectron.SOLVER.IMS_PER_BATCH = image_per_batch
    detectron.SOLVER.BASE_LR = lr
    detectron.SOLVER.MAX_ITER = max_iter
    detectron.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = batch_size
    detectron.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    detectron.OUTPUT_DIR = output_dir
    detectron.TEST.EVAL_PERIOD = eval_period

    os.makedirs(detectron.OUTPUT_DIR, exist_ok=True)
    if use_augment:
        trainer = AugmentTrainer(detectron)
    else:
        trainer = CustomTrainer(detectron)

    trainer.resume_or_load(resume=False)
    trainer.train()

def setup_config(config_path):
    cfg = get_config()
    cfg.merge_from_file(config_path)

    return cfg

if __name__ == '__main__':
    cfg = setup_config('configs/config.yaml')
    # get config
    use_json = cfg.CONFIG.USE_JSON 
    train_json_path = cfg.CONFIG.TRAIN_JSON_PATH 
    val_json_path = cfg.CONFIG.VAL_JSON_PATH 

    use_xml = cfg.CONFIG.USE_XML 
    root_dir = cfg.CONFIG.ROOT_DIR 
    train_dir = cfg.CONFIG.TRAIN_DIR 
    label_train_dir = cfg.CONFIG.LABEL_TRAIN_DIR 
    image_train_dir = cfg.CONFIG.IMAGE_TRAIN_DIR
    val_dir = cfg.CONFIG.VAL_DIR 
    label_val_dir = cfg.CONFIG.LABEL_VAL_DIR 
    image_val_dir = cfg.CONFIG.IMAGE_VAL_DIR 
    json_val_name = cfg.CONFIG.JSON_VAL_NAME 
    json_train_name = cfg.CONFIG.JSON_TRAIN_NAME 

    list_classes = load_class_names(cfg.CONFIG.LABELS_PATH)

    use_augment = cfg.CONFIG.USE_AUGMENT 
    size = cfg.CONFIG.SIZE 
    flip_prob = cfg.CONFIG.FLIP_PROB 
    min_brightness = cfg.CONFIG.MIN_BRIGHTNESS 
    max_brightness = cfg.CONFIG.MAX_BRIGHTNESS 
    min_contrast = cfg.CONFIG.MIN_CONTRAST 
    max_contrast = cfg.CONFIG.MAX_CONTRAST 
    min_saturation = cfg.CONFIG.MIN_SATURATION 
    max_saturation = cfg.CONFIG.MAX_SATURATION 

    config_model_path = cfg.CONFIG.CONFIG_MODEL_PATH 
    num_workers = cfg.CONFIG.NUM_WORKERS
    weights_path = cfg.CONFIG.WEIGHTS_PATH 
    image_per_batch = cfg.CONFIG.IMAGE_PER_BATCH 
    lr = cfg.CONFIG.LR 
    max_iter = cfg.CONFIG.MAX_ITER 
    batch_size = cfg.CONFIG.BATCH_SIZE 
    num_classes = cfg.CONFIG.NUM_CLASSES
    output_dir = cfg.CONFIG.OUTPUT_DIR
    eval_period = cfg.CONFIG.EVAL_PERIOD

    # check use json
    if use_json:
        train_dicts, val_dicts = read_json(train_json_path, val_json_path)
    else:
        train_dicts, val_dicts = create_json(root_dir, train_dir, label_train_dir, image_train_dir, \
                                        val_dir, label_val_dir, image_val_dir, json_val_name, json_train_name, use_xml)
    
    # regist data
    dataset_train, dataset_val = regist_data(root_dir, train_dicts, val_dicts, list_classes)

    # training
    detectron = config_detectron()
    Training(detectron, config_model_path, dataset_train, dataset_val, num_workers, weights_path, image_per_batch, \
        lr, max_iter, batch_size, num_classes, output_dir, eval_period, use_augment)