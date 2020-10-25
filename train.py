import os
import sys

from utils.utils import load_class_names
from utils.parser import get_config

sys.path.append("Efficientdet/lib/")
from train_detector import Detector

def train(detector, class_list, root_dir, train_dir, img_train_dir, set_train_dir, \
        val_dir, img_val_dir, set_val_dir, model_name, num_gpus, freeze_head, \
        optimizer, lr, es_min_delta, es_patience, batch_size, \
        num_workers, num_epochs, val_interval, save_interval):
    
    # setup train dataset
    detector.set_train_dataset(root_dir, train_dir, img_train_dir, set_train_dir, \
                            classes_list=class_list, batch_size=batch_size, num_workers=num_workers)
    
    # setup val dataset
    detector.set_val_dataset(root_dir, val_dir, img_val_dir, set_val_dir)

    # setup model
    detector.set_model(model_name="efficientdet-d0.pth", num_gpus=num_gpus, freeze_head=freeze_head)

    # setup optimizer
    detector.set_hyperparams(optimizer=optimizer, lr=lr, es_min_delta=es_min_delta, es_patience=es_patience)

    # training
    detector.train(num_epochs=num_epochs, val_interval=val_interval, save_interval=save_interval)

def setup_config(config_path):
    cfg = get_config()
    cfg.merge_from_file(config_path)

    return cfg

if __name__ == '__main__':
    # get config
    cfg = setup_config('config.yaml')
    root_dir = cfg.CONFIG.ROOT_DIR 
    class_list = load_class_names(cfg.CONFIG.LABELS_PATH)
    train_dir = cfg.CONFIG.TRAIN_DIR 
    img_train_dir = cfg.CONFIG.IMG_TRAIN_DIR 
    set_train_dir = cfg.CONFIG.SET_TRAIN_DIR 
    val_dir = cfg.CONFIG.VAL_DIR 
    img_val_dir = cfg.CONFIG.IMG_VAL_DIR 
    set_val_dir = cfg.CONFIG.SET_VAL_DIR 
    model_name = cfg.CONFIG.MODEL_NAME 
    num_gpus = cfg.CONFIG.NUM_GPUS 
    freeze_head = cfg.CONFIG.FREEZE_HEAD 
    optimizer = cfg.CONFIG.OPTIMIZER 
    lr = cfg.CONFIG.LR 
    es_min_delta = cfg.CONFIG.ES_MIN_DELTA 
    es_patience = cfg.CONFIG.ES_PATIENCE 
    batch_size = cfg.CONFIG.BATCH_SIZE 
    num_workers = cfg.CONFIG.NUM_WORKERS 
    num_epochs = cfg.CONFIG.NUM_EPOCHS 
    val_interval = cfg.CONFIG.VAL_INTERVAL 
    save_interval = cfg.CONFIG.SAVE_INTERVAL 

    # get detector
    detector = Detector()

    # start training
    train(detector, class_list, root_dir, train_dir, img_train_dir, set_train_dir, \
        val_dir, img_val_dir, set_val_dir, model_name, num_gpus, freeze_head, \
        optimizer, lr, es_min_delta, es_patience, batch_size, \
        num_workers, num_epochs, val_interval, save_interval)
    