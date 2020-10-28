import os
import cv2
import json
import random
import itertools
import numpy as np
import cv2

from time import gmtime, strftime
from utils.parser import get_config
from utils.utils import load_class_names

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg as config_detectron

def predict(image, predictor, list_labels):
    outputs = predictor(image)

    boxes = outputs['instances'].pred_boxes
    scores = outputs['instances'].scores
    classes = outputs['instances'].pred_classes

    list_boxes = []
    list_scores = []
    list_classes = []

    for i in range(len(classes)):
        if (scores[i] > 0.6):
            for j in boxes[i]:
                x = int(j[0])
                y = int(j[1])
                w = int(j[2]) - x
                h = int(j[3]) - y

            score = float(scores[i])
            class_id = list_labels[int(classes[i])]

            list_boxes.append([x, y, w, h])
            list_scores.append(score)
            list_classes.append(class_id)
    
    return list_boxes, list_scores, list_classes

if __name__ == '__main__':
    cfg = get_config()
    cfg.merge_from_file('configs/detect.yaml')

    path_weigth = cfg.DETECTOR.DETECT_WEIGHT
    path_config = cfg.DETECTOR.DETECT_CONFIG
    confidences_threshold = cfg.DETECTOR.THRESHOLD
    num_of_class = cfg.DETECTOR.NUMBER_CLASS

    detectron = config_detectron()
    detectron.MODEL.DEVICE= cfg.DETECTOR.DEVICE
    detectron.merge_from_file(path_config)
    detectron.MODEL.WEIGHTS = path_weigth

    detectron.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidences_threshold
    detectron.MODEL.ROI_HEADS.NUM_CLASSES = num_of_class

    PREDICTOR = DefaultPredictor(detectron)
    CLASSES = load_class_names(cfg.DETECTOR.VEHICLE_CLASS)

    image = cv2.imread("demo.jpg")
    list_boxes, list_scores, list_classes = predict(image, PREDICTOR, CLASSES)
    print("list_classes: ", list_classes)