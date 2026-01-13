import cv2
import numpy as np

class ColorDetection:
    def __init__(self, lower_bound : np.ndarray, upper_bound : np.ndarray, grid_rows : int, grid_cols : int):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
    
    def findMask(self, img : np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_bound, self.upper_bound)
        kernel = np.ones((5, 5), np.uint8)

        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        processed_mask = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

        return processed_mask

    def detect(self, frame : np.ndarray):
        mask = self.findMask(frame)
        h, w = mask.shape

        # Creating a grid.
        grid_h = h // self.grid_rows
        grid_w = w // self.grid_rows
        density_map = np.zeros((self.grid_rows, self.grid_cols), dtype=np.int32)

        # Populating the density map.
        for i in range(self.grid_rows):
            for j in range(self.grid_cols):
                cell = mask[i * grid_h:(i+1) * grid_h, j * grid_w:(j+1) * grid_w]
                density_map[i, j] = cv2.countNonZero(cell)

        # Find the grid cell with the most detected color.
        max_idx = np.unravel_index(np.argmax(density_map), density_map.shape)
        i_max, j_max = max_idx

        # Calculating the pixel centroid.
        x_center = int(j_max * grid_w + grid_w / 2)
        y_center = int(i_max * grid_h + grid_h / 2)

        # Calculating error relative to the center of the frame.
        x_error = x_center - w // 2
        y_error = h // 2 - y_center

        # Outputting debug frame with overlay.
        debug_frame = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(debug_frame, (j_max * grid_w, i_max * grid_h), ((j_max+1) * grid_w, (i_max+1) * grid_h), (0, 0, 255), 2)
        cv2.circle(debug_frame, (x_center, y_center), 5, (0, 255, 0), -1)
        
        return x_error, y_error, debug_frame