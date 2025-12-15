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

    def find_angle(self, contour):
        if contour is None or len(contour) < 2:
            return None
        
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        lines = []
        for i in range(len(approx)):
            p1 = approx[i][0]
            p2 = approx[(i + 1) % len(approx)][0]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = math.hypot(dx, dy)
            if length > 2:
                angle = math.degrees(math.atan2(dy, dx))
                lines.append((length, angle, p1, p2))

        if len(lines) < 2:
            return None
        
        lines.sort(reverse=True, key=lambda x: x[0])
        longest = lines[:2]

        angles = [math.radians(l[1]) for l in longest]
        x_mean = np.mean([math.cos(a) for a in angles])
        y_mean = np.mean([math.sin(a) for a in angles])
        avg_angle = math.degrees(math.atan2(y_mean, x_mean))

        avg_angle = (avg_angle + 360 + 90) % 180
        return avg_angle

    def detect(self, frame):
        if self.backend == "pt":
            results = self.model.predict(source=frame, conf=self.conf_thresh, verbose=False)

            if not results or len(results[0].boxes) == 0:
                return None, 0.0, frame
            
            boxes = results[0].boxes
            index = int(boxes.conf.argmax())
            best = boxes[index]
            
            x1, y1, x2, y2 = best.xyxy[0].cpu().numpy().astype(int)
            confidence = float(best.conf.cpu().numpy())

        elif self.backend == "rknn":
            img, scale, left, top = self.letterbox(frame)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            outputs = self.rknn.inference(inputs=[img_rgb])
            detections = outputs[0] # Todo: Doublecheck output. [N, 5] -> x1, y1, x2, y2, confidence.

            if detections is None or len(detections) == 0:
                return None, 0.0, frame
            
            detections = np.array(detections)
            best_index = detections[:, 4].argmax()
            x1, y1, x2, y2, confidence = detections[best_index]

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

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None, 0.0, frame
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contour = max(contours, key=cv2.contourArea) if contours else None
        angle = self.find_angle(contour)

        debug_frame = frame.copy()
        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        rect = np.array([x1, y1, w, h, angle], dtype=np.float32)

        return rect, confidence, debug_frame