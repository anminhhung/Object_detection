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