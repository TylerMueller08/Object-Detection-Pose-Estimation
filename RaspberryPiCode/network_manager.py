import ntcore
from cscore import CameraServer
from cv2 import Mat
import time

class NetworkManager:
    """
    A class used to manage the network connection and camera setup for a robot.
    Attributes
    ----------
    team_number : int
        The team number of the robot, used to determine the robot's IP address.

    robot_ip : str
        The IP address of the robot, formatted as "team_number.local".

    nt : ntcore.NetworkTableInstance
        The NetworkTable instance used for communication.

    data_table : ntcore.NetworkTable
        The NetworkTable used to store and retrieve data.
        
    camera_publisher : cscore.VideoSink
        The publisher for the camera stream.

    topics : dict
        A dictionary of topics for publishing data to the NetworkTable.

    publishers : dict
        A dictionary of publishers for the topics defined in `topics`.

    Methods
    -------
    setup_camera(camera_name: str)
        Sets up the camera with the specified name and resolution.
    publish_image(image: Mat)
        Publishes an image to the camera stream.
    setup_topics()
        Sets up the topics for publishing data to the NetworkTable.
    game_piece_position(x: float, y: float, a: float, certainty: float = 0.0)
        Publishes the robot's position and certainty to the NetworkTable.
    """

    def __init__(self, team_number : int):
        self.robot_ip = str(team_number) + ".local"  # Assuming the robot's IP is in the format "team_number.local".
        
        print("Initializing NetworkManager")
        print("Configured Robot IP:", self.robot_ip)
        self.nt = ntcore.NetworkTableInstance.getDefault()

        table_name = "vision"
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
        """ Sets up the camera on the robot. """
        self.camera_publisher = CameraServer.putVideo(camera_name, 640, 480) # Set the resolution to 640x480.
        self.camera_publisher.setFPS(30) # Limit CPU usage and bandwidth.
        print("Camera Stream Connected")
    
    def publish_image(self, image : Mat):
        """ Publishes an image to the camera stream. """
        self.camera_publisher.putFrame(image)

    def setup_topics(self):
        self.topics = {
            "game_piece_position_x" : self.data_table.getDoubleTopic("game_piece_position_x"),
            "game_piece_position_y" : self.data_table.getDoubleTopic("game_piece_position_y"),
            "game_piece_yaw" : self.data_table.getDoubleTopic("game_piece_yaw"),
            "certainty" : self.data_table.getDoubleTopic("certainty")
        }

        self.publishers = {
            "game_piece_position_x" : self.topics["game_piece_position_x"].publish(),
            "game_piece_position_y" : self.topics["game_piece_position_y"].publish(),
            "game_piece_yaw" : self.topics["game_piece_yaw"].publish(),
            "certainty" : self.topics["certainty"].publish()
        }        

    def publish_game_piece_position(self, x, y, a, certainty=0.0):
        """ Publishes the game piece's position to the NetworkTable. """
        timestamp = ntcore._now()
        self.publishers["game_piece_position_x"].set(x, timestamp)
        self.publishers["game_piece_position_y"].set(y, timestamp)
        self.publishers["game_piece_yaw"].set(a, timestamp)
        self.publishers["certainty"].set(certainty, timestamp)