from ultralytics import YOLO
import cv2

model = YOLO('../Yolo-Weights/yolov8n.pt')
results = model("images/3.jpg",show=True)
cv2.waitKey(0)