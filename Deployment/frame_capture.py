import time
import cv2

class FrameCapture:
    def __init__(self, camera_id : int, camera_resolution : tuple[int, int], target_fps : int):
        self.camera_id = camera_id
        print("Initializing Camera with ID:", camera_id)
        self.cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, target_fps)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','J','P','G'))
    
    def capture_frame(self) -> tuple[cv2.Mat, float]:
        while True:
            start_time = time.time()
            ret, frame = self.cap.read()
            if ret:
                # print("FPS:", 1 / (time.time() - start_time))
                return (frame, start_time)
