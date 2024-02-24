import os
import cv2
import time
import numpy as np
import onnxruntime

from .utils.augment import letterbox
from .utils.nms import nms, xywh2xyxy


class YoloORT:
    def __init__(self, names:list=None, weight:str=None, img_size:tuple=None, conf:float=0.2, iou:float=0.1) -> None:
        
        self.names:list = names
        rng = np.random.default_rng(3)
        self.colors = rng.uniform(0, 255, size=(len(self.names), 3))
        
        self.conf_threshold:float = conf
        self.iou_threshold:float = iou
        self.new_shape:float = img_size
    
        # init model
        self.model_initialize(model_path=weight)
    
    def model_initialize(self, model_path:str) -> None:
        self.session = onnxruntime.InferenceSession(
            model_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        
        # get input details
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height, self.input_width = self.input_shape[2], self.input_shape[3]
        
        # get output details
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]
    
    def detect(self, image:np.ndarray):
        self.start_time = time.perf_counter()
        input_tensor, letterbox_image = self.preprocess(image=image)
        letterbox_image = cv2.cvtColor(letterbox_image, cv2.COLOR_RGB2BGR)
        detect_outputs = self.inference(input_tensor=input_tensor)

        result = self.process_outputs(outputs=detect_outputs)
        del input_tensor
        infer_time_usage = time.perf_counter() - self.start_time
        return result  #, letterbox_image, input_tensor
        
    def preprocess(self, image:np.ndarray):
        letterbox_image, self.scale, (self.letter_left, self.letter_top) = letterbox(
            im=image,
            new_shape=self.new_shape,
            auto=False
        )
        self.image_height, self.image_width = letterbox_image.shape[:2]
        letterbox_image = cv2.cvtColor(letterbox_image, cv2.COLOR_BGR2RGB)
        input_image = letterbox_image / 255.0
        # del letterbox_image
        input_image = input_image.transpose(2, 0, 1)
        input_tensor = input_image[np.newaxis, :, :, :].astype(np.float32) 
        return input_tensor, letterbox_image
    
    def inference(self, input_tensor):
        outputs = self.session.run(
            self.output_names,
            {self.input_names[0]: input_tensor}
        )
        return outputs
    
    def process_outputs(self, outputs):
        
        num_detects = int(outputs[0][0][0])
        detect_bbox = outputs[1].tolist()[0]
        detect_score = outputs[2].tolist()[0]
        detect_class = outputs[3].tolist()[0]
        
        if num_detects == 0:
            return []
        
        result = []
        for bbox_coord, conf, cls in zip(detect_bbox[:num_detects-1], detect_score[:num_detects-1], detect_class[:num_detects-1]):
            if int(cls) == -1:
                continue
            object_conf = float(conf)
            if object_conf < self.conf_threshold:
                continue
            if int(cls) >= len(self.names):
                continue
            x0, y0, x1, y1 = bbox_coord
            
            x0, y0, x1, y1 = x0-self.letter_left, y0-self.letter_top, x1-self.letter_left, y1-self.letter_top
            x0, y0, x1, y1 = int(x0/self.scale), int(y0/self.scale), int(x1/self.scale), int(y1/self.scale)
            
            single_hand = [int(cls), x0, y0, x1, y1, object_conf]
            result.append(single_hand)
        
        return result
           
        