## Project Details
Pipeline based on Yet-Another-EfficientDet project - https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch  


<br />
<br />
<br />

# Supported Models
  - efficientdet-d0.pth
  - efficientdet-d1.pth
  - efficientdet-d2.pth
  - efficientdet-d3.pth
  - efficientdet-d4.pth
  - efficientdet-d5.pth
  - efficientdet-d6.pth
  - efficientdet-d7.pth

<br />
<br />

## Installation

Supports 
- Python 3.6
- Cuda 9.0, 10.0 (Other cuda version support is experimental)
    
`cd installation`

`cat requirements_cuda9.0.txt | xargs -n 1 -L 1 pip install`

<br />
<br />
<br />



## Pipeline

- Load Dataset

`gtf.set_train_dataset(root_dir, coco_dir, img_dir, set_dir, classes_list=["class1", "class2"], batch_size=2, num_workers=3)`

- Load Model

`gtf.set_model(model_name="efficientdet-d3.pth", num_gpus=1, freeze_head=False);`

- Set Hyper Parameters

`gtf.set_hyperparams(optimizer="adamw", lr=0.001, es_min_delta=0.0, es_patience=0)`

- Train

`gtf.train(num_epochs=20, val_interval=1, save_interval=1)`



<br />
<br />
<br />

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

