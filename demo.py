import cv2
import numpy
from YoloORT import YoloORT


camera_index:int = 0
capture = cv2.VideoCapture(camera_index)

class_names:list = ['standby', 'one', 'ok', 'two_up', 'two up inverted', 'peace', 'peace inverted','stop','stop inverted']
onnx_model:str = "/home/kd/Documents/Codes/YoloV6_Onnx/20_ckpt.onnx"
detect_backend = YoloORT(weight=onnx_model, img_size=(640, 640), names=class_names, conf=0.3)

while True:
    ret, frame = capture.read()
    if not ret:
        raise ValueError("No Frame!")
    
    result = detect_backend.detect(image=frame)
    for id, x0, y0, x1, y1, conf in result:
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 1, 1)
        cv2.putText(frame, class_names[id] + f" {conf:.2f}", (x0, y0), 1, 1, (0, 0, 255), 1, 1)
    
    cv2.imshow(f"Frame", frame)
    cv2.waitKey(1)


