import cv2
from YoloORT import YoloORT

# set camera
camera_index:int = 0
capture = cv2.VideoCapture(camera_index)

# set model
class_names:list = ['1', '2', '3']
onnx_model:str = "{Your Model Path}/model.onnx"

# init model
detect_backend = YoloORT(
    weight=onnx_model, 
    img_size=(640, 640), 
    cls_name=class_names, 
    conf=0.3, device="cuda"
)

while True:
    ret, frame = capture.read()
    if not ret:
        raise ValueError("No Frame!")
    
    result:list = detect_backend.detect(image=frame)  # detect
    
    if result is None:
        continue
    
    fps = detect_backend.get_fps()
    cv2.putText(frame, f"{fps:.2f} FPS", (20, 20), 1, 1, (0, 0, 255), 1, 1)
    for id, x0, y0, x1, y1, conf in result:
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 1, 1)
        cv2.putText(frame, class_names[id] + f" {conf:.2f}", (x0, y0), 1, 1, (0, 0, 255), 1, 1)
    
    cv2.imshow(f"Frame", frame)
    cv2.waitKey(1)