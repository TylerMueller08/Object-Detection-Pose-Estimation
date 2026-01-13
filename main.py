import queue
import numpy as np
import multiprocessing as mp
from frame_capture import FrameCapture
from color_detection import ColorDetection
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

def detection_process(frame_queue : mp.Queue, detection_queue : mp.Queue, debug_queue : mp.Queue, lower_bound : np.ndarray, upper_bound : np.ndarray):
    detector = ColorDetection(lower_bound, upper_bound, grid_rows=12, grid_cols=16)
    while True:
        try:
            frame, timestamp = frame_queue.get(timeout=0.01)
        except queue.Empty:
            continue

        x_error, y_error, debug_frame = detector.detect(frame)

        try:
            detection_queue.put(((x_error, y_error), timestamp), timeout=0.01)
            if debug_queue is not None:
                debug_queue.put(debug_frame)
        except queue.Full:
            pass

def network_management_process(debug_queue : mp.Queue, detection_queue : mp.Queue, team_number : int, simulation : bool, debug_stream : bool):
    network_manager = NetworkManager(team_number, simulation, debug_stream)
    while True:
        if debug_stream:
            try:
                frame = debug_queue.get(timeout=0.01)
                network_manager.publish_image(frame)
            except queue.Empty:
                pass

        try:
            error, timestamp = detection_queue.get(timeout=0.01)
            network_manager.publish_game_piece_position(error, timestamp)
        except queue.Empty:
            pass

def main():
    # Camera Resolution & Settings.
    camera_id = 0
    camera_resolution = (1280, 720)
    camera_fps = 60

    # Calibrate bounds.
    lower_bound = np.array([20, 100, 100])
    upper_bound = np.array([35, 255, 255])

    # Team Number for NetworkTable.
    team_number = 4593
    simulation = True
    debug_stream = True

    frame_queue = mp.Queue(maxsize=5)
    detection_queue = mp.Queue(maxsize=5)
    debug_queue = mp.Queue(maxsize=5)

    processes = [
        mp.Process(target=frame_capture_process, args=(frame_queue, camera_id, camera_resolution, camera_fps)),
        mp.Process(target=detection_process, args=(frame_queue, detection_queue, debug_queue, lower_bound, upper_bound)),
        mp.Process(target=network_management_process, args=(debug_queue, detection_queue, team_number, simulation, debug_stream))
    ]

    for process in processes:
        process.start()

    for process in processes:
        process.join()

if __name__ == "__main__":
    main()