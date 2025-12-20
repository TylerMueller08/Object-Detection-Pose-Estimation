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
        if frame is None:
            break
        try:
            frame_queue.put((frame, timestamp), timeout=0.01)
        except queue.Full:
            pass

def detection_process(frame_queue : mp.Queue, detection_queue : mp.Queue, debug_queue : mp.Queue, model_path : str, backend : str):
    detector = ObjectDetection(model_path, backend)
    while True:
        try:
            frame, timestamp = frame_queue.get(timeout=0.01)
        except queue.Empty:
            continue

        rect, confidence, debug_frame = detector.detect(frame)

        try:
            detection_queue.put((rect, confidence, timestamp), timeout=0.01)
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
    
        position = None
        if rect is not None:
            position = estimator.estimate_position(rect)

            try:
                position_queue.put((position, confidence, timestamp), timeout=0.01)
            except queue.Full:
                pass

def network_management_process(debug_queue : mp.Queue, position_queue : mp.Queue, team_number : int, simulation : bool, debug_stream : bool):
    network_manager = NetworkManager(team_number, simulation, debug_stream)
    while True:
        if debug_stream:
            try:
                frame = debug_queue.get(timeout=0.01)
                network_manager.publish_image(frame)
            except queue.Empty:
                pass

        try:
            position, confidence, timestamp = position_queue.get(timeout=0.01)
            network_manager.publish_game_piece_position(position, confidence, timestamp)
        except queue.Empty:
            pass

def main():
    # Camera Resolution & Settings.
    camera_id = 0
    camera_resolution = (1280, 720)
    camera_fps = 30

    # Train YOLO model.
    model_path = "resources/Coral-640-640-yolov11n.pt"
    backend = "pt" # Choose "pt" or "rknn".

    # Team Number for NetworkTable.
    team_number = 4593
    simulation = True
    debug_stream = True

    frame_queue = mp.Queue(maxsize=1)
    detection_queue = mp.Queue(maxsize=1)
    position_queue = mp.Queue(maxsize=1)
    debug_queue = mp.Queue(maxsize=1)

    # Load estimator data (example data).
    estimator_data = pd.read_csv('resources/CoralData.csv') # Todo: Run with desired camera configuration.

    processes = [
        mp.Process(target=frame_capture_process, args=(frame_queue, camera_id, camera_resolution, camera_fps)),
        mp.Process(target=detection_process, args=(frame_queue, detection_queue, debug_queue, model_path, backend)),
        mp.Process(target=position_estimation_process, args=(detection_queue, position_queue, estimator_data)),
        mp.Process(target=network_management_process, args=(debug_queue, position_queue, team_number, simulation, debug_stream))
    ]

    for process in processes:
        process.start()

    for process in processes:
        process.join()

if __name__ == "__main__":
    main()