## Project Details
Pipeline based on facebookresearch - detectron2: [Detectron2](https://github.com/facebookresearch/detectron2)

---

## Installation
`pip3 install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.6/index.html`  
`pip3 install -r requirements.txt`  

---

## Dataset Directory Structure

        ../dataset (root_dir)
            |
            |------val (val_dir) 
            |      |
            |      |----images (img_val_dir)
            |      |    |
            |      |    |---------img1.jpg 
            |      |    |---------img2.jpg 
            |      |    |---------..........(and so on)
            |      |
            |      |----labels (label_val_dir) (xml or txt)
            |      |    |
            |      |    |---------label1 
            |      |    |---------label2 
            |      |    |---------..........(and so on)
            |      |    
            |      |-----classes.txt
            |
            |------train (train_dir) 
            |      |
            |      |----images (img_train_dir)
            |      |    |
            |      |    |---------img1.jpg 
            |      |    |---------img2.jpg 
            |      |    |---------..........(and so on)
            |      |
            |      |----labels (label_train_dir) (xml or txt)
            |      |    |
            |      |    |---------label1 
            |      |    |---------label2 
            |      |    |---------..........(and so on)
            |      |    
            |      |-----classes.txt

---

## Create json 
- path of train file: root_dir/train_dir/<json_name>.json  
- path of val file: root_dir/val_dir/<json_name>.json

---

## Data augmentation method
Detectron2 has a large list of [available data augmentation methods](https://github.com/facebookresearch/detectron2/tree/master/detectron2/data/transforms).  

- **RandomApply**: This is wrapper around the other aumentation methods so that you can turn a list of them on or off as a group with a specified probability.  
- **Resize and ResizeShortestEdge**: The are two resize options. If your images vary in shape and you don't want to distort them, *ResizeShortestEdge* is the one to use because it won't change the image's aspect ratio.  
- **RandomRotation**: This does exactly what it sounds like. You pass it a list of *[min_angle, max_angle]* and it randomly chooses a value. This can be useful for overhead imagery because your object are usually rotation-invariant. If you rotate an image, it will by default preserve all the information in the original image by adding black padding to all the corners. Therefore the actual image size increases, and if you're not-rotated image batch fills up your GPU, rotating the image can cause it to run out of memory. To prevent this, you can do *RandomRotation(45, expand=False). However, this will clip the corners of your original image, so you will lose information.  
- **RandomContrast, RandomBrightness, and RandomSaturation**: These are all pretty straightforward. You can provide them ranges of the form *(min, max)* where the value 1 is an identity function (no change). These augmentations are very important in cases where different classes might have variation in one of these.  

**Implementation**:  
```python
from detectron2.data import transforms as T
train_augmentations = [
    T.Resize((200, 200)),
    T.RandomBrightness(0.5, 1.5),
    T.RandomContrast(0.5, 1.5),
    T.RandomSaturation(0.5, 1.5),
    T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
    T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
]
```

Modify [here](https://github.com/anminhhung/Object_detection/blob/2dcdce992ff39106536c3037fb3bdb5fec810a74/Detectron/train.py#L93).

---

## Setup hyper params
```python
CONFIG:
    # Use JSON
    USE_JSON:   # True if use json format, otherwise
    TRAIN_JSON_PATH:    # Path of train json file or None 
    VAL_JSON_PATH:      # Path of val json file or None

    # Use XML OR TXT
    USE_XML:    # True if use xml format, otherwise
    ROOT_DIR:   
    TRAIN_DIR:
    LABEL_TRAIN_DIR:
    IMAGE_TRAIN_DIR:
    VAL_DIR:
    LABEL_VAL_DIR:
    IMAGE_VAL_DIR:
    JSON_VAL_NAME:
    JSON_TRAIN_NAME:

    # File labels
    LABELS_PATH: 

    # Use Augment
    USE_AUGMENT:    # True if use, otherwise
    SIZE:   # use for resize
    FLIP_PROB:      # use for random flip
    MIN_BRIGHTNESS:    # use for brigtness
    MAX_BRIGHTNESS:     # use for brightness
    MIN_CONTRAST:    # use for contrast
    MAX_CONTRAST:    # use for contrast
    MIN_SATURATION:     # use for saturation
    MAX_SATURATION:     # use for saturation

    # Training
    CONFIG_MODEL_PATH:
    NUM_WORKERS:
    WEIGHTS_PATH:
    IMAGE_PER_BATCH:
    LR:
    MAX_ITER:
    BATCH_SIZE:
    NUM_CLASSES:
    OUTPUT_DIR:     # path of model
    EVAL_PERIOD:
```

--- 

## Run

        python3 train.py