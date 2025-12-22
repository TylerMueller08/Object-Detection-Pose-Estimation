import numpy as np
import cv2, math

class ObjectDetection:
    def __init__(self, model_path, backend="pt", conf_thresh=0.4, input_size=(640, 640)):
        self.backend = backend
        self.conf_thresh = conf_thresh
        self.input_size = input_size

        if backend == "pt":
            try:
                from ultralytics import YOLO
            except ImportError:
                raise RuntimeError("Error: PyTorch YOLO cannot be installed.")
            self.model = YOLO(model_path)
        elif backend == "rknn":
            try:
                from rknnlite.api import RKNNLite
            except ImportError:
                raise RuntimeError("Error: RKNN Lite cannot be installed.")
            self.rknn = RKNNLite()
            self.rknn.load_rknn(model_path)
            self.rknn.init_runtime()
        else:
            raise ValueError('Backend must be either "pt" or "rknn"')

    def letterbox(self, image):
        h, w = image.shape[:2]
        target_w, target_h = self.input_size

        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(image, (new_w, new_h))
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

        top = (target_h - new_h) // 2
        left = (target_w - new_w) // 2
        canvas[top:top + new_h, left:left + new_w] = resized

        return canvas, scale, left, top

    def detect(self, frame):
        if self.backend == "pt":
            results = self.model.predict(source=frame, conf=self.conf_thresh, verbose=False)

            if not results or len(results[0].boxes) == 0:
                return None, frame
            
            boxes = results[0].boxes
            index = int(boxes.conf.argmax())
            best = boxes[index]
            
            x1, y1, x2, y2 = best.xyxy[0].cpu().numpy().astype(int)

        elif self.backend == "rknn":
            img, scale, left, top = self.letterbox(frame)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            outputs = self.rknn.inference(inputs=[img_rgb])
            detections = outputs[0] # Todo: Doublecheck output. [N, 5] -> x1, y1, x2, y2.

            if detections is None or len(detections) == 0:
                return None, frame
            
            detections = np.array(detections)
            best_index = detections[:, 4].argmax()
            x1, y1, x2, y2 = detections[best_index]

            x1 = int((x1 - left) / scale)
            y1 = int((y1 - top) / scale)
            x2 = int((x2 - left) / scale)
            y2 = int((y2 - top) / scale)

        else:
            raise RuntimeError("Error: Unsupported backend logic.")
        
        w = x2 - x1
        h = y2 - y1

        h_img, w_img = frame.shape[:2]

        x1 = max(0, min(x1, w_img - 1))
        y1 = max(0, min(y1, h_img - 1))
        x2 = max(0, min(x2, w_img))
        y2 = max(0, min(y2, h_img))

        debug_frame = frame.copy()
        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        rect = np.array([x1, y1, w, h], dtype=np.float32)

        return rect, debug_frame