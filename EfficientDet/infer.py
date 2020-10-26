from lib import Infer
import cv2

detector = Infer()

# model_path
model_path = "efficientdet-d0_trained.pth"

# class_list
classes_list = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14"]

detector.load_model(model_path, classes_list, use_gpu=False)

image_path = "vehicle_detect.png"
image, scores, labels, bboxes = detector.predict(image_path, one_image=True, threshold=0.5, imshow=False, imwrite=False)

cv2.imshow("image", image)
cv2.waitKey(0)