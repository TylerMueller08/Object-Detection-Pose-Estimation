import ntcore
from cscore import CameraServer
from cv2 import Mat
import time

class NetworkManager:
    def __init__(self, team_number : int):
        self.robot_ip = str(team_number) + ".local" # Robot's IP is "team_number.local", or "127.0.0.1" if simulation.
        
        print("Initializing NetworkManager")
        print("Configured Robot IP:", self.robot_ip)
        self.nt = ntcore.NetworkTableInstance.getDefault()

        table_name = "ObjectDetection"
        self.data_table = self.nt.getTable(table_name)
        print(f"Connected to NetworkTables: '{table_name}'")

        self.setup_camera("DebugCamera")
        print("Established Camera")

        self.setup_topics()
        print("Established NetworkTables Topics")

        self.nt.startClient4("orangepi5_" + str(team_number))
        self.nt.setServerTeam(team_number) # or self.nt.setServer("127.0.0.1") if simulaton.
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
            "objectConfidence" : self.data_table.getDoubleTopic("objectConfidence")
        }

        self.publishers = {
            "objectPose" : self.topics["objectPose"].publish(),
            "objectConfidence" : self.topics["objectConfidence"].publish()
        }        

    def publish_game_piece_position(self, x_position, y_position, angle, confidence):
        timestamp = ntcore._now()
        self.publishers["objectPose"].set([x_position, y_position, angle], timestamp)
        self.publishers["objectConfidence"].set(confidence, timestamp)