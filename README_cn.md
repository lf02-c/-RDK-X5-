# 隧安哨兵——基于 RDK X5 的多传感融合隧道智能巡检机器人

## 项目简介

隧安哨兵是一套面向隧道巡检场景设计的智能巡检机器人。系统基于 **RDK X5 AI 开发板**，融合激光雷达、摄像头和 STM32 底盘控制器，结合 **ROS 2、Nav2、自主导航、YOLO 目标检测、多传感融合** 等技术，实现隧道环境下的自主巡检与智能预警。

## 项目特色

- 激光雷达 SLAM 建图
- ROS 2 Nav2 自主导航
- YOLO + RDK X5 BPU 加速推理
- 火灾检测
- 人员检测
- 安全门状态检测
- 实时语音报警
- 自动生成巡检报告
- Web 上位机管理

## 硬件平台

- RDK X5 AI 开发板
- LDLiDAR LD14P
- USB 摄像头
- STM32 底盘控制器
- 差速移动机器人

## 软件平台

- Ubuntu 22.04
- ROS 2 Humble
- Nav2
- OpenCV
- YOLO
- Python
- C++

## 项目目录

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

## 核心模块说明

| 文件 / 目录 | 功能说明 |
|---|---|
| maps/ | 存放 SLAM 地图、导航配置、禁区数据及 Keepout 掩码。 |
| models/ | YOLO 模型文件。 |
| door_detect_node.py | 图像采集、BPU 推理、目标检测与结果发布。 |
| patrol_node.py | 巡检任务控制、自主导航、巡检结果汇总。 |
| alarm_node.py | 语音报警与异常提示。 |
| stm32_bridge_node | ROS2 与 STM32 底盘通信。 |
| ldlidar | LD14P 激光雷达驱动。 |
| navigation2.launch.py | 系统总启动文件。 |
| web_bridge_node | Web 地图、巡检点、禁区管理。 |
| keyboard_control_node | 建图阶段键盘遥控。 |
| keepout_mask | 自动生成 Nav2 禁区掩码。 |
| slam_with_keyboard.launch.py | 建图启动文件。 |
| map_saver_cli | 保存地图。 |

## 系统流程

激光雷达 → SLAM 建图 → AMCL 定位 → Nav2 导航 → 巡检点 → YOLO 检测 → 多传感融合 → 语音报警 → 巡检报告

## 应用场景

- 公路隧道
- 铁路隧道
- 地下通道
- 地下停车场
- 地下综合管廊

## 许可证

本项目仅用于科研、教学及全国大学生嵌入式芯片与系统设计竞赛交流。
