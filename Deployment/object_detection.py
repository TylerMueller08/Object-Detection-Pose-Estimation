import cv2
import numpy as np
import math
from rknnlite.api import RKNNLite

class ObjectDetection:
    def __init__(self, rknn_path, input_size=(640, 640), conf_thresh=0.4):
        self.input_size = input_size
        self.conf_thresh = conf_thresh
        self.rknn = RKNNLite()

        ret = self.rknn.load_rknn(rknn_path)
        if ret != 0:
            raise RuntimeError("Error: Failed to load RKNN model.")
        ret = self.rknn.init_runtime()
        if ret != 0:
            raise RuntimeError("Error: Failed to initialize RKNN runtime.")
        
    def preprocess(self, frame):
        # img = cv2.resize(frame, self.input_size) # Stretches, not letterboxing (fix if needed).
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        return img

    def postprocess(self, outputs, orig_shape):
        h0, w0, _ = orig_shape
        detections = outputs[0]

        if detections is None or len(detections) == 0:
            return None
        
        best = max(detections, key=lambda d: d[4])
        if best[4] < self.conf_thresh:
            return None
        
        cx, cy, w, h = best[:4]
        cx *= w0
        cy *= h0
        w *= w0
        h *= h0

        x = int(cx - w / 2)
        y = int(cy - h / 2)

        return np.array([x, y, int(w), int(h)], dtype=np.int32)
    
    def find_angle(self, contour):
        if len(contour) < 2:
            return 0.0
        
        # 1. Simplify the contour (removing small jitter).
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # 2. Collect line segments and their angles.
        lines = []
        for i in range(len(approx)):
            p1 = approx[i][0]
            p2 = approx[(i + 1) % len(approx)][0] # Wrap around.
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = math.hypot(dx, dy)
            if length > 2: # Ignore tiny edges.
                angle = math.degrees(math.atan2(dy, dx))
                lines.append((length, angle, p1, p2))

        if len(lines) < 2:
            return None
        
        # 3. Take two longest lines.
        lines.sort(reverse=True, key=lambda x: x[0])
        longest = lines[:2]

        # 4. Compute average direction.
        # Handle circular mean (avoid averaging 179° and -179° to get 0°)
        angles = [math.radians(l[1]) for l in longest]
        x_mean = np.mean([math.cos(a) for a in angles])
        y_mean = np.mean([math.sin(a) for a in angles])
        avg_angle = math.degrees(math.atan2(y_mean, x_mean))

        avg_angle = (avg_angle + 360 + 90) % 180 # Normalize to [0, 180).
        return avg_angle

    def detect(self, frame):
        inp = self.preprocess(frame)
        outputs = self.rknn.inference(inputs=[inp])

        rect = self.postprocess(outputs, frame.shape)
        if rect is None:
            return None
        
        x, y, w, h = rect
        roi = frame[y:y+h, x:x+w]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contour = max(contours, key=cv2.contourArea) if contours else None
        angle = self.find_angle(contour)

        return np.array([x, y, w, h, angle], dtype=np.float32)