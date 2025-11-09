# UR5 CoppeliaSim ROS Integration

Vision-based surface detection and robot control system for UR5 in CoppeliaSim using ROS 2.

## Overview

This project uses depth sensor data to detect circular features on surfaces and control a UR5 robot arm in CoppeliaSim. The system performs two scanning passes with different detection strategies and adjusts the robot's position accordingly.

## Requirements

- Python 3.x
- ROS 2
- CoppeliaSim
- Dependencies:
  - numpy
  - scipy
  - open3d
  - scikit-learn
  - scikit-image
  - rclpy
  - zmq
  - msgpack
  - coppeliasim_zmqremoteapi_client

Install Python dependencies:
```bash
pip install numpy scipy open3d scikit-learn scikit-image pyzmq msgpack
```

## Components

### vision_depth_publisher.py
Publishes depth sensor data from CoppeliaSim's vision sensor via ZMQ on port 24000.

### first_sweep.py
First scanning pass that:
- Collects 840 depth images while sweeping the sensor
- Segments planes from point cloud data
- Detects circular features using least squares fitting
- Publishes detected surface poses to ROS topic `surface_pose`

Tunable parameters at the top of the file include scan steps, FOV, thresholds for plane detection, and circle radius bounds.

### adjust_dummy_1.py
ROS node that subscribes to `surface_pose` and moves the UR5 target dummy in CoppeliaSim. Uses smooth interpolation with SLERP for rotations. Applies a -180 degree offset to the Z-axis orientation.

### second_sweep.py
Second scanning pass with more advanced detection:
- Uses RANSAC for line fitting
- Detects orientation from line direction on circular features
- Includes retry logic (up to 10 attempts with Z-offset adjustments)
- Saves detection results and visualizes point clouds with Open3D
- Output saved to `output_data/` directory

### adjust_dummy_2.py
Similar to adjust_dummy_1 but without the Z-axis orientation adjustment. Uses raw quaternion from incoming pose.

### sequence.py
Orchestrates execution of multiple scripts:
1. Starts drum manipulator (drum_manipulator.py - not included)
2. Runs first sweep and adjustment
3. Waits for movement completion
4. Runs second sweep and adjustment

## Usage

### Single Pass Detection

Run the vision publisher first:
```bash
python vision_depth_publisher.py
```

In separate terminals:
```bash
python first_sweep.py
python adjust_dummy_1.py
```

Or for the second sweep:
```bash
python second_sweep.py
python adjust_dummy_2.py
```

### Full Sequence

Make sure `drum_manipulator.py` exists, then:
```bash
python sequence.py
```

## Configuration

Each script has tunable parameters at the top:

**Scanning parameters:**
- `NUM_STEPS`: Number of scan positions
- `STEP_DELAY_TOTAL`: Total time for scan movement
- Scan range Y offsets

**Detection parameters:**
- `MAX_PLANES`: Maximum planes to segment
- `PLANE_DISTANCE_THRESHOLD`: RANSAC threshold for plane fitting
- `DBSCAN_EPS`: Clustering epsilon
- `CIRCLE_RADIUS_MIN/MAX`: Valid circle radius bounds

**Motion parameters:**
- `SMOOTH_TRANSFORM_DURATION`: Time for interpolated movement
- `SMOOTH_TRANSFORM_STEPS`: Interpolation steps
- `STAGE1_PAUSE`: Pause between movement stages

## ROS Topics

- `surface_pose` (geometry_msgs/Pose): Detected surface position and orientation
- `surface_euler` (geometry_msgs/Vector3): Euler angles of detected surface (first sweep only)

## Notes

- CoppeliaSim must be running with a UR5 model that includes `/UR5/target` dummy and `/UR5/visionSensor`
- The vision sensor should have a 120-degree horizontal FOV
- ZMQ communication runs on localhost:24000
- Second sweep saves visualization data to `output_data/` directory with incremental numbering

## Coordinate Systems

The system uses CoppeliaSim's world frame. Transformation pipeline:
1. Depth image to camera frame point cloud
2. Transform to world frame using sensor pose
3. Plane segmentation and feature detection in world frame
4. Publish target poses relative to UR5 parent frame

