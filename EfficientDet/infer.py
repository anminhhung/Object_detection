from lib import Infer
from IPython.display import Image

detector = Infer()

# model_path
model_path = 

# class_list
classes_list = 

detector.load_model(model_path, classes_list, use_gpu=True)

image_path = 
detector.predict(img_path, threshold=0.5)

Image(filename='output.jpg')