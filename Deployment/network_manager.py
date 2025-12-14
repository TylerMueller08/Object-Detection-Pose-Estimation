import ntcore
from cscore import CameraServer
from cv2 import Mat
import time

class NetworkManager:
    def __init__(self, team_number : int):
        self.robot_ip = str(team_number) + ".local" # Robot's IP is "team_number.local".
        
        print("Initializing NetworkManager")
        print("Configured Robot IP:", self.robot_ip)
        self.nt = ntcore.NetworkTableInstance.getDefault()

        table_name = "ObjectDetection"
        self.data_table = self.nt.getTable(table_name)
        print(f"Connected to NetworkTables: '{table_name}'")

        self.setup_camera("Camera")
        print("Established Camera")

        self.setup_topics()
        print("Established NetworkTables Topics")

        self.nt.startClient4("orangepi5_" + str(team_number))
        self.nt.setServerTeam(team_number, 0)
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
            "objectPositionX" : self.data_table.getDoubleTopic("objectPositionX"),
            "objectPositionY" : self.data_table.getDoubleTopic("objectPositionY"),
            "objectAngle" : self.data_table.getDoubleTopic("objectAngle"),
            "objectConfidence" : self.data_table.getDoubleTopic("objectConfidence")
        }

        self.publishers = {
            "objectPositionX" : self.topics["objectPositionX"].publish(),
            "objectPositionY" : self.topics["objectPositionY"].publish(),
            "objectAngle" : self.topics["objectAngle"].publish(),
            "objectConfidence" : self.topics["objectConfidence"].publish()
        }        

    def publish_game_piece_position(self, x, y, a, certainty=0.0):
        timestamp = ntcore._now()
        self.publishers["objectPositionX"].set(x, timestamp)
        self.publishers["objectPositionY"].set(y, timestamp)
        self.publishers["objectAngle"].set(a, timestamp)
        self.publishers["objectConfidence"].set(certainty, timestamp)