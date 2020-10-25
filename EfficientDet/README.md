## Project Details
Pipeline based on Yet-Another-EfficientDet project - [EfficientDet](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch  )  
and Monk Object Detection - [Monk_Object_Detection](https://github.com/anminhhung/Monk_Object_Detection).

---

## Supported Models
  - efficientdet-d0.pth
  - efficientdet-d1.pth
  - efficientdet-d2.pth
  - efficientdet-d3.pth
  - efficientdet-d4.pth
  - efficientdet-d5.pth
  - efficientdet-d6.pth
  - efficientdet-d7.pth

---

## Available optimizers
  - adamw
  - sgd

---

## Installation

Supports 
- Python 3.6
- Cuda 9.0, 10.0 (Other cuda version support is experimental)
    
`cd installation`  
`cat requirements_cuda9.0.txt | xargs -n 1 -L 1 pip install`

Use Colab:  
!cd Efficientdet/installation/ && cat requirements_colab.txt | xargs -n 1 -L 1 pip install

---

## Dataset Directory Structure (Format 1)

    ../sample_dataset (root_dir)
          |
          |------dataset (coco_dir) 
          |         |
          |         |----images (img_dir)
          |                |
          |                |------Train (set_dir) (Train)
          |                         |
          |                         |---------img1.jpg
          |                         |---------img2.jpg
          |                         |---------..........(and so on)
          |                |-------Val (set_dir) (Validation)
          |                         |
          |                         |---------img1.jpg
          |                         |---------img2.jpg
          |                         |---------..........(and so on)  
          |
          |
          |         |---annotations 
          |         |----|
          |              |--------------------instances_Train.json  (instances_<set_dir>.json)
          |              |--------------------instances_Val.json  (instances_<set_dir>.json)
          |              |--------------------classes.txt
          
          
 - instances_Train.json -> In proper COCO format
 - classes.txt          -> A list of classes in alphabetical order
 

For TrainSet
 - root_dir = "../sample_dataset";
 - coco_dir = "dataset";
 - img_dir = "images";
 - set_dir = "Train";
 
For ValSet
 - root_dir = "..sample_dataset";
 - coco_dir = "dataset";
 - img_dir = "images";
 - set_dir = "Val";
 
 Note: Annotation file name too coincides against the set_dir

---

## Dataset Directory Structure (Format 2)

    ../sample_dataset (root_dir)
          |
          |------dataset (coco_dir) 
          |         |
          |         |---ImagesTrain (set_dir)
          |         |----|
          |              |-------------------img1.jpg
          |              |-------------------img2.jpg
          |              |-------------------.........(and so on)
          |
          |
          |         |---ImagesVal (set_dir)
          |         |----|
          |              |-------------------img1.jpg
          |              |-------------------img2.jpg
          |              |-------------------.........(and so on)
          |
          |
          |         |---annotations 
          |         |----|
          |              |--------------------instances_ImagesTrain.json  (instances_<set_dir>.json)
          |              |--------------------instances_ImagesVal.json  (instances_<set_dir>.json)
          |              |--------------------classes.txt
          
          
 - instances_Train.json -> In proper COCO format
 - classes.txt          -> A list of classes in alphabetical order
 
 For TrainSet
 - root_dir = "../sample_dataset";
 - coco_dir = "dataset";
 - img_dir = "./";
 - set_dir = "ImagesTrain";
 
 
  For ValSet
 - root_dir = "../sample_dataset";
 - coco_dir = "dataset";
 - img_dir = "./";
 - set_dir = "ImagesVal";
 
 Note: Annotation file name too coincides against the set_dir
 

---

## Setup hyper params

```python
CONFIG:
    # setup dataset
    ROOT_DIR: 
    LABELS_PATH:
    TRAIN_DIR:
    IMG_TRAIN_DIR:
    SET_TRAIN_DIR:
    VAL_DIR:
    IMG_VAL_DIR:
    SET_VAL_DIR:
    
    # Avalabel models
    MODEL_NAME:
    
    # hyper params
    NUM_GPUS:
    FREEZE_HEAD:
    OPTIMIZER:
    LR:
    ES_MIN_DELTA:
    ES_PATIENCE:
    BATCH_SIZE:
    NUM_WORKERS:
    NUM_EPOCHS:
    VAL_INTERVAL:
    SAVE_INTERVAL:
```

---

## Model
- Link model: [**LINK**](https://drive.google.com/drive/u/0/folders/1CxcGUwtdWYOLbqAEtBbMQ63cEVNzbBjB)

---

## Run
    python3 train.py

---

## TODO

- [x] Add support for Coco-Type Annotated Datasets
- [x] Add support for VOC-Type Annotated Dataset
- [x] Test on Kaggle and Colab 
- [x] Add validation feature & data pipeline
- [x] Add Optimizer selection feature
- [x] Enable Layer Freezing
- [x] Set Verbosity Levels
- [x] Add Multi-GPU training


<br />
<br />
<br />

