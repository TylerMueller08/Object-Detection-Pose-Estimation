import numpy as np
import pandas as pd

class PoseEstimation:
    def __init__(self, width: int, height: int, data : pd.DataFrame):
        self.width = width
        self.height = height
        self.data = data
    
    def find_matching_rows(self, df, target, start_tol=25, max_tol=40, step=3) -> tuple[pd.DataFrame, int]:
        tolerance = start_tol
        while tolerance <= max_tol:
            # Create mask using np.isclose for each column.
            mask = (
                (np.isclose(df['Center_X'], target['Center_X'], atol=tolerance) &
                np.isclose(df['Center_Y'], target['Center_Y'], atol=tolerance) &
                np.isclose(df['Width'], target['Width'], atol=tolerance) &
                np.isclose(df['Height'], target['Height'], atol=tolerance)) &
                (np.isclose(df['Angle'], target['Angle'], atol=tolerance) |
                    np.isclose(df['Angle'], (target['Angle'] + 180) % 360, atol=tolerance) |
                    np.isclose(df['Angle'], (target['Angle'] - 180) % 360, atol=tolerance))
            )
            filtered_df = df[mask]
            
            # Check if any row is found.
            if not filtered_df.empty:
                print(f"Found Rows w/ Tolerance: {tolerance}")
                return filtered_df, tolerance
            
            # Increase the tolerance and try again.
            tolerance += step

        # No rows found within maximum tolerance.
        # print("No Rows Within Maximum Tolerance.")
        return df.iloc[[]], tolerance  # Return an empty DataFrame.
    
    def estimate_position(self, rectangle: np.ndarray) -> tuple[float, float, float, int]:
        x, y, w, h, a = rectangle
        center_x = x + w // 2
        center_y = y + h // 2
        
        target = {
            'Center_X': center_x,
            'Center_Y': center_y,
            'Width': w,
            'Height': h,
            'Angle': a
        }
        
        matching_rows, used_tol = self.find_matching_rows(self.data, target, 25, 40, 3)
        
        if not matching_rows.empty:
            return (matching_rows['x_position'].mean(), matching_rows['y_position'].mean(), matching_rows['angle'].mean(), (50 - used_tol))
        
        # If no match is found, return None for position and 0 for certainty.
        return None