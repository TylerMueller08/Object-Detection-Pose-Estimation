# Object Detection Pose Estimation

*A fast, robust, and field-ready algorithm for real-time object pose estimation in FIRST Robotics Competition (FRC) adapted from [YoavRozov](https://github.com/YoavRozov/FRC-Game-Piece-Pos-Estimation).*

---
## Documentation
Full setup instructions, Blender configuration, and usage examples are available in the Step-by-Step Guide.

**Read the full setup guide**:
  [Here](https://yoavrozov.github.io/FRC-Game-Piece-Pos-Estimation/setup/)

---
## Overview
This project provides a complete solution for estimating the real-world pose (position and orientation) of FRC Game Pieces from camera frames using an optimized, pre-trained machine learning model. It uses a data-driven image-matching approach, offering extremely fast and accurate pose estimation—optimized for Orange Pi 5 deployment in competition.

## Key Features

- **High-Speed & Accuracy**  
  Matches live camera images to a precomputed dataset of rendered frames for sub-10ms pose estimation.
  
- **Modular, Multi-Process Architecture**  
  Ensures real-time performance by offloading detection, estimation, and networking to separate processes.

- **Built for FRC Robotics**  
  Seamless integration with FRC systems via **NetworkTables**, optimized for the Orange Pi 5 or similar Rockchip systems (SoCs) with a built-in Neural Processing Unit (NPU)

---

## How It Works

### 1. Preprocessing (One-Time, Offline)
- Render thousands of labeled images in **Blender** using the provided scripts.
- Each frame encodes the 3D pose of the game piece from a fixed camera viewpoint.
- Store the dataset in CSV format with pose metadata.

### 2. Runtime (On-Robot)
- Capture live frames from the robot’s fixed-position camera.
- Detect the game piece using machine learning segmentation.
- Extract bounding rectangles and (optionally) orientation info.
- Find the closest match in the dataset using a **fast image similarity search**.
- Output the real-world position and orientation of the game piece.
