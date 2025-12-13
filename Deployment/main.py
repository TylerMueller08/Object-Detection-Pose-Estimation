import multiprocessing as mp
import pandas as pd
from frame_capture import FrameCapture
from object_detection import ObjectDetection
from pose_estimation import PoseEstimation
from network_manager import NetworkManager

def frame_capture_process(queue : mp.Queue, camera_id : int, resolution : tuple[int, int], fps : int):
    frame_capture = FrameCapture(camera_id, resolution, fps)
    while True:
        frame, timestamp = frame_capture.capture_frame()
        if not queue.full():
            queue.put((frame, timestamp))

def detection_process(queue : mp.Queue, detection_queue : mp.Queue, rknn_path : str):
    detector = ObjectDetection(rknn_path)
    while True:
        if not queue.empty():
            frame, timestamp = queue.get()
            rect = detector.detect(frame)
            if rect is None:
                continue
            if not detection_queue.full():
                detection_queue.put((rect, timestamp))

def position_estimation_process(detection_queue : mp.Queue, position_queue : mp.Queue, estimator_data : pd.DataFrame):
    estimator = PoseEstimation(1280, 720, estimator_data)
    while True:
        if not detection_queue.empty():
            rect, timestamp = detection_queue.get()
            position = estimator.estimate_position(rect)
            if position is None:
                continue
            if not position_queue.full():
                position_queue.put((position, timestamp))

def network_management_process(position_queue : mp.Queue, team_number : int):
    network_manager = NetworkManager(team_number)
    while True:
        if not position_queue.empty():
            position, timestamp = position_queue.get()
            x, y, angle, certainty = position
            network_manager.publish_game_piece_position(x, y, angle, certainty)
        else:
            # If no position is available, you can publish a default value or wait.
            network_manager.publish_game_piece_position(0, 0, 0, 0) # Spam, needs fixing?

def main():
    # Camera Resolution & Settings.
    camera_id = 0
    camera_resolution = (1280, 720)
    camera_fps = 30

    # Train YOLO model.
    rknn_path = "Coral-640-640-yolov11n.rknn"

    # Team Number for NetworkTable.
    team_number = 4593

    frame_queue = mp.Queue(maxsize=1)
    detection_queue = mp.Queue(maxsize=1)
    position_queue = mp.Queue(maxsize=1)

    # Load estimator data (example data).
    estimator_data = pd.read_csv('../Data/2025-Coral/FullData.csv') # Need to run ourselves.

    processes = [
        mp.Process(target=frame_capture_process, args=(frame_queue, camera_id, camera_resolution, camera_fps)),
        mp.Process(target=detection_process, args=(frame_queue, detection_queue, rknn_path)),
        mp.Process(target=position_estimation_process, args=(detection_queue, position_queue, estimator_data)),
        mp.Process(target=network_management_process, args=(position_queue, team_number))
    ]

    for process in processes:
        process.start()

    for process in processes:
        process.join()

if __name__ == "__main__":
    main()