# Tunnel Sentinel: Multi-Sensor Fusion Intelligent Tunnel Inspection Robot Based on RDK X5

## Overview

Tunnel Sentinel is an intelligent autonomous inspection robot designed for tunnel safety monitoring. Built on the RDK X5 AI computing platform, the system integrates LiDAR SLAM, autonomous navigation, computer vision, and multi-sensor fusion technologies to perform autonomous patrol tasks in tunnel environments.

The robot supports autonomous mapping, localization, path planning, target detection, and real-time voice warning, providing an efficient and intelligent solution for tunnel inspection and safety management.

## Features

- Autonomous mapping using LiDAR SLAM
- Autonomous localization and navigation based on ROS 2 Nav2
- YOLO-based real-time object detection with RDK X5 BPU acceleration
- Multi-sensor fusion perception
- Fire detection
- Human intrusion detection
- Safety door status detection
- Real-time voice alarm
- Automatic patrol report generation
- Web-based patrol management interface

## Hardware Platform

- RDK X5 AI Development Board
- LDLiDAR LD14P
- USB Camera
- Differential Drive Mobile Robot
- STM32 Motor Controller

## Software Stack

- Ubuntu 22.04
- ROS 2 Humble
- Nav2
- OpenCV
- YOLO
- Python
- C++

## Project Structure

```text
LD/
├── config/
├── launch/
├── maps/
├── models/
├── rviz/
├── scripts/
├── src/
└── README.md
```

## Core Modules

| Directory / File | Description |
|---|---|
| maps/ | SLAM maps (.yaml/.pgm), navigation configuration, keep-out zone data and masks. |
| models/ | YOLO models for fire, human and safety door detection. |
| door_detect_node.py | Camera capture, OpenCV preprocessing, BPU inference and detection result publishing. |
| patrol_node.py | Patrol controller, waypoint management, autonomous navigation and patrol reporting. |
| alarm_node.py | Voice alarm and warning notifications. |
| stm32_bridge_node | Converts ROS 2 `/cmd_vel` into STM32 serial commands and handles odometry. |
| ldlidar | LD14P LiDAR driver for SLAM and navigation. |
| navigation2.launch.py | Starts LiDAR, Nav2 stack, STM32 bridge and navigation nodes. |
| web_bridge_node | Browser-based map visualization, waypoint editing and patrol management. |
| keyboard_control_node | Manual keyboard teleoperation during mapping. |
| keepout_mask | Generates Nav2 keep-out mask files. |
| slam_with_keyboard.launch.py | Mapping launch file. |
| map_saver_cli | Saves generated maps. |

## Workflow

LiDAR → SLAM → AMCL → Nav2 → Patrol → Camera + YOLO → Multi-sensor Fusion → Voice Alarm → Patrol Report

## Application Scenarios

- Highway tunnels
- Railway tunnels
- Underground passages
- Underground parking garages
- Utility corridors

## License

This project is developed for academic research and the National College Embedded Chip and System Design Competition.
