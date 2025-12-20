import ntcore
from cscore import CameraServer
from cv2 import Mat
import time, math

class NetworkManager:
    def __init__(self, team_number : int, simulation : bool, debug_stream : bool):
        self.robot_ip = "127.0.0.1" if simulation else str(team_number) + ".local"
        
        print("Initializing NetworkManager")
        print("Configured Robot IP:", self.robot_ip)
        
        self.nt = ntcore.NetworkTableInstance.getDefault()

        table_name = "ObjectDetection"
        self.data_table = self.nt.getTable(table_name)

        print(f"Connected to NetworkTables: '{table_name}'")

        if debug_stream:
            self.setup_camera("ObjectDetection")
            print("Established Camera")

        self.setup_topics()
        print("Established NetworkTables Topics")

        self.nt.startClient4("orangepi5_" + str(team_number))

        if simulation:
            self.nt.setServer("127.0.0.1")
        else:
            self.nt.setServerTeam(team_number)

        time.sleep(3)
        print("NetworkTables Client Started")

    def setup_camera(self, camera_name):
        self.camera_publisher = CameraServer.putVideo(camera_name, 640, 480) # Set the resolution to 640x480.
        self.camera_publisher.setFPS(30) # Limit CPU usage and bandwidth.
        print("Camera Stream Connected")
    
    def publish_image(self, image : Mat):
        self.camera_publisher.putFrame(image)

    def setup_topics(self):
        self.topics = {
            "objectPose" : self.data_table.getDoubleArrayTopic("objectPose"),
            "objectConfidence" : self.data_table.getDoubleTopic("objectConfidence"),
            "objectLatencySec" : self.data_table.getDoubleTopic("objectLatencySec")
        }

        self.publishers = {
            "objectPose" : self.topics["objectPose"].publish(),
            "objectConfidence" : self.topics["objectConfidence"].publish(),
            "objectLatencySec" : self.topics["objectLatencySec"].publish()
        }        

    def publish_game_piece_position(self, position, confidence, capture_timestamp):
        publish_time = time.monotonic()
        latency = publish_time - capture_timestamp

        nt_timestamp = int(capture_timestamp * 1e6)

        if position is None:
            pose = [math.nan, math.nan, math.nan]
            confidence = 0.0
        else:
            x, y, angle = position
            pose = [x, y, angle]

        self.publishers["objectPose"].set(pose, nt_timestamp)
        self.publishers["objectConfidence"].set(confidence, nt_timestamp)
        self.publishers["objectLatencySec"].set(latency, nt_timestamp)