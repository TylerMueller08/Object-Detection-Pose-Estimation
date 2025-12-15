import numpy as np
import pandas as pd

class PoseEstimation:
    def __init__(self, width: int, height: int, data : pd.DataFrame):
        self.width = width
        self.height = height
        self.data = data
    
    def find_matching_rows(self, df, target, start_tol=25, max_tol=40, step=3):
        tolerance = start_tol
        while tolerance <= max_tol:
            mask = (
                np.isclose(df['x_center'], target['Center_X'], atol=tolerance) &
                np.isclose(df['y_center'], target['Center_Y'], atol=tolerance) &
                np.isclose(df['width'], target['Width'], atol=tolerance) &
                np.isclose(df['height'], target['Height'], atol=tolerance) &
                (
                    np.isclose(df['Image_angle'], target['Angle'], atol=tolerance) |
                    np.isclose(df['Image_angle'], (target['Angle'] + 180) % 360, atol=tolerance) |
                    np.isclose(df['Image_angle'], (target['Angle'] - 180) % 360, atol=tolerance)
                )
            )
            filtered = df[mask]
            if not filtered.empty:
                return filtered
            tolerance += step
        return df.iloc[[]]
    
    def estimate_position(self, rectangle: np.ndarray):
        x, y, w, h, a = rectangle
        center_x = x + w / 2
        center_y = y + h / 2

        target = {
            'Center_X': center_x,
            'Center_Y': center_y,
            'Width': w,
            'Height': h,
            'Angle': a
        }
        matching_rows = self.find_matching_rows(self.data, target)

        if not matching_rows.empty:
            return (
                matching_rows['x_position'].mean(),
                matching_rows['y_position'].mean(),
                matching_rows['angle'].mean()
            )
        
        return None