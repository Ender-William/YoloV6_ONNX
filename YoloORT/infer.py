import os
import cv2
import time
import numpy as np
import onnxruntime

from .utils.augment import letterbox

class YoloORT:
    def __init__(self, cls_name:list=None, weight:str=None, img_size:tuple=(640, 640), 
                 conf:float=0.2, device:str="cuda") -> None:
        """Init YoloV6 ORT Detect Model

        Args:
            cls_name (list, optional): class name. Defaults to None.
            weight (str, optional): model path. Defaults to None.
            img_size (tuple, optional): model input shape. Defaults to (640, 640).
            conf (float, optional): conf threshold. Defaults to 0.2.
            device (str, optional): equipment used for inference. Default to "cuda", support ["cuda", "cpu"].
        """
        self.names:list = cls_name
        self.conf_threshold:float = conf
        self.new_shape:float = img_size
        
        if weight is None:
            raise ValueError("No models loaded")
        weight_exists = os.path.exists(weight)
        if not weight_exists:
            raise ValueError("Model is not exist")        
    
        # init model
        self.set_ort_provider(device=device)
        self.model_initialize(model_path=weight)
    
    def set_ort_provider(self, device:str):
        if device == "trt":
            self.provider = ['TensorrtExecutionProvider']
        elif device == "cuda":
            self.provider = ['CUDAExecutionProvider']
        elif device == "cpu":
            self.provider = ['CPUExecutionProvider']
        else:
            print(f"device: {device} is not support, use cpu providers")
            self.provider = ['CPUExecutionProvider']
    
    def model_initialize(self, model_path:str) -> None:
        self.session = onnxruntime.InferenceSession(
            model_path,
            providers=self.provider
        )
        
        # get input details
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height, self.input_width = self.input_shape[2], self.input_shape[3]
        
        # get output details
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]
    
    def detect(self, image:np.ndarray = None) -> tuple:
        
        if image is None: return None
        
        self.start_time = time.perf_counter()
        
        input_tensor = self.preprocess(image=image)
        detect_outputs = self.inference(input_tensor=input_tensor)

        result = self.process_outputs(outputs=detect_outputs)
        del input_tensor
        del detect_outputs
        self.infer_time_usage = time.perf_counter() - self.start_time
        self.frame_per_second = 1 / self.infer_time_usage
        return result
    
    def get_fps(self) -> float:
        return self.frame_per_second
    
    def get_infer_time(self) -> float:
        return self.infer_time_usage
        
    def preprocess(self, image:np.ndarray):
        """get letterbox image and tensor image

        Args:
            image (np.ndarray): bgr image

        Returns:
            NDArray: tensor image
        """
        letterbox_image, self.scale, (self.letter_left, self.letter_top) = letterbox(
            im=image,
            new_shape=self.new_shape,
            auto=False
        )
        self.image_height, self.image_width = letterbox_image.shape[:2]
        letterbox_image = cv2.cvtColor(letterbox_image, cv2.COLOR_BGR2RGB)
        input_image = letterbox_image / 255.0
        del letterbox_image
        input_image = input_image.transpose(2, 0, 1)
        input_tensor = input_image[np.newaxis, :, :, :].astype(np.float32) 
        del input_image
        return input_tensor
    
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
           
        