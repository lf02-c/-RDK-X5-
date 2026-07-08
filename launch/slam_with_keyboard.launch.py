#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    # ========== launch ==========
    ldlidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('ldlidar'),
                'launch',
                'ld14p.launch.py'
            )
        )
    )

    # ========== SLAM Toolbox ==========
    slam_node = Node(
        package='slam_toolbox',
        executable='sync_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        parameters=[{
            'use_sim_time': False
        }]
    )

    # ========== RVIZ ==========
    pkg_share = get_package_share_directory('autopatrol_robot')

    rviz_config = os.path.join(
        pkg_share,
        'config',
        'slam_with_keyboard.rviz'
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config]
    )

    return LaunchDescription([
        ldlidar_launch,
        slam_node,
        rviz_node
    ])
