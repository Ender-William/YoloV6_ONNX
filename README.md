# YoloV6 Onnxruntime Inference Pack

此版本的 YoloV6 在 Onnxruntime 的推理程序，训练模型需要使用 KKPIP-Tech 组织 fork 的 YoloV6 版本，[链接在此](https://github.com/KKPIP-Tech/YOLOv6)。或使用官方的 0.4.1 版本进行训练。

## Environment Requirements
```
Python >= 3.10

pip install onnxruntime-gpu
pip install opencv-python
pip install numpy
```
在使用 CUDA 推理时，需要在本机配置好 CUDA 以及 cuDNN。如果需要使用 TensorRT，还需要自行配置 TRT 相关组件。

## Export for ONNX
在 YoloV6 的项目中，使用下面的指令导出 onnx 模型
```shell
python deploy/ONNX/export_onnx.py \
    --weights {Your Weight Path} \
    --img 640 \
    --batch 1 \
    --end2end \
    --ort \
    --conf 0.3 \
    --iou 0.1
```

## Build
```shell
python setup.py sdist bdist_wheel
```

## Demonstration
```python
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
```

## Reference

> [YoloV6 meituan](https://github.com/meituan/YOLOv6)

> [YoloV6 KKPIP](https://github.com/KKPIP-Tech/YOLOv6)

> [ONNX-YOLOv6-Object-Detection](https://github.com/ibaiGorordo/ONNX-YOLOv6-Object-Detection)
