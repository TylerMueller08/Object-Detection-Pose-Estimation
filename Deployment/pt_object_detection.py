from ultralytics import YOLO
import numpy as np
import cv2, math

class PtObjectDetection:
    def __init__(self, model_path, conf_thresh=0.4):
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh

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
        results = self.model.predict(source=frame, conf=self.conf_thresh, verbose=False)

        if not results or len(results[0].boxes) == 0:
            return None
        
        boxes = results[0].boxes
        best = boxes[boxes.conf.argmax()]
        
        x1, y1, x2, y2 = best.xyxy[0].cpu().numpy().astype(int)
        w = x2 - x1
        h = y2 - y1

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contour = max(contours, key=cv2.contourArea) if contours else None
        angle = self.find_angle(contour)

        return np.array([x1, y1, w, h, angle], dtype=np.float32)