import torch
import numpy as np

class PoseEstimation:
    def __init__(self, model_path: str):
        from regression_model import PoseMLP
        self.device = torch.device("cpu")
        self.model = PoseMLP()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def estimate_position(self, rectangle: np.ndarray):
        if rectangle is None or any(np.isnan(rectangle)):
            return None

        x, y, w, h = rectangle

        img_w, img_h = 1280, 720
        cx = (x + w / 2) / img_w
        cy = (y + h / 2) / img_h
        w_norm = w / img_w
        h_norm = h / img_h

        input_data = torch.tensor([[cx, cy, w_norm, h_norm]], dtype=torch.float32).to(self.device)

        with torch.no_grad():
            pred = self.model(input_data)
            world_x, world_y, world_yaw = pred[0].cpu().numpy().tolist()

        return world_x, world_y, world_yaw