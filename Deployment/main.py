import multiprocessing as mp
import queue
import pandas as pd
from frame_capture import FrameCapture
from object_detection import ObjectDetection
from pose_estimation import PoseEstimation
from network_manager import NetworkManager

def frame_capture_process(frame_queue : mp.Queue, camera_id : int, resolution : tuple[int, int], fps : int):
    frame_capture = FrameCapture(camera_id, resolution, fps)
    while True:
        frame, timestamp = frame_capture.capture_frame()
        try:
            frame_queue.put((frame, timestamp), timeout=0.01)
        except queue.Full:
            pass

def detection_process(frame_queue : mp.Queue, detection_queue : mp.Queue, debug_queue : mp.Queue, model_path : str):
    detector = ObjectDetection(model_path)
    while True:
        try:
            frame, timestamp = frame_queue.get(timeout=0.01)
        except queue.Empty:
            continue

        rect, confidence, debug_frame = detector.detect(frame)

        try:
            if rect is not None:
                detection_queue.put((rect, confidence, timestamp))
            if debug_queue is not None:
                debug_queue.put(debug_frame)
        except queue.Full:
            pass

def position_estimation_process(detection_queue : mp.Queue, position_queue : mp.Queue, estimator_data : pd.DataFrame):
    estimator = PoseEstimation(1280, 720, estimator_data)
    while True:
        try:
            rect, confidence, timestamp = detection_queue.get(timeout=0.01)
        except queue.Empty:
            continue
    
        position = estimator.estimate_position(rect)
        if position is not None:
            try:
                position_queue.put((position, confidence, timestamp), timeout=0.01)
            except queue.Full:

                pass

def network_management_process(debug_queue : mp.Queue, position_queue : mp.Queue, team_number : int):
    network_manager = NetworkManager(team_number)
    while True:
        try:
            frame = debug_queue.get(timeout=0.01)
            network_manager.publish_image(frame)
        except queue.Empty:
            pass

        try:
            position, confidence, timestamp = position_queue.get(timeout=0.01)
            x, y, angle = position
            network_manager.publish_game_piece_position(x, y, angle, confidence)
        except queue.Empty:
            pass

def main():
    # Camera Resolution & Settings.
    camera_id = 0
    camera_resolution = (1280, 720)
    camera_fps = 30

    # Train YOLO model.
    model_path = "resources/Coral-640-640-yolov11n.pt"

    # Team Number for NetworkTable.
    team_number = 4593

    frame_queue = mp.Queue(maxsize=1)
    detection_queue = mp.Queue(maxsize=1)
    position_queue = mp.Queue(maxsize=1)
    debug_queue = mp.Queue(maxsize=1)

    # Load estimator data (example data).
    estimator_data = pd.read_csv('../Data/2025-Coral/FullData.csv') # Need to run ourselves.

    processes = [
        mp.Process(target=frame_capture_process, args=(frame_queue, camera_id, camera_resolution, camera_fps)),
        mp.Process(target=detection_process, args=(frame_queue, detection_queue, debug_queue, model_path)),
        mp.Process(target=position_estimation_process, args=(detection_queue, position_queue, estimator_data)),
        mp.Process(target=network_management_process, args=(debug_queue, position_queue, team_number))
    ]

    for process in processes:
        process.start()

    for process in processes:
        process.join()

if __name__ == "__main__":
    main()