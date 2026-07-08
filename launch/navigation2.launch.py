#!/usr/bin/env python3
import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from nav2_common.launch import RewrittenYaml


def _launch_boolean(value):
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _map_image_path(yaml_path):
    for raw_line in yaml_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        if key.strip() != "image":
            continue
        image_value = value.strip().strip("'\"")
        if not image_value:
            break
        image_path = Path(image_value).expanduser()
        if image_path.is_absolute():
            return image_path
        return yaml_path.parent / image_path
    raise RuntimeError(f"keepout mask YAML 缺少 image 字段：{yaml_path}")


def _validate_keepout_configuration(context):
    enabled_value = LaunchConfiguration("enable_keepout").perform(context)
    if not _launch_boolean(enabled_value):
        return []

    map_value = LaunchConfiguration("map").perform(context).strip()
    mask_value = LaunchConfiguration("keepout_mask").perform(context).strip()
    if not mask_value:
        raise RuntimeError(
            "enable_keepout=true 时必须提供 keepout_mask YAML 路径"
        )

    map_path = Path(map_value).expanduser()
    mask_path = Path(mask_value).expanduser()
    if not mask_path.is_file():
        raise RuntimeError(f"keepout mask YAML 不存在：{mask_path}")
    expected_mask_stem = f"{map_path.stem}_keepout"
    if mask_path.stem != expected_mask_stem:
        raise RuntimeError(
            "keepout mask 与原地图名称不匹配："
            f"期望 {expected_mask_stem}.yaml，实际 {mask_path.name}"
        )

    mask_image_path = _map_image_path(mask_path)
    if not mask_image_path.is_file():
        raise RuntimeError(f"keepout mask PGM 不存在：{mask_image_path}")

    zones_path = map_path.parent / f"{map_path.stem}_zones.json"
    if zones_path.is_file():
        mask_timestamp = min(
            mask_path.stat().st_mtime_ns,
            mask_image_path.stat().st_mtime_ns,
        )
        if mask_timestamp < zones_path.stat().st_mtime_ns:
            raise RuntimeError(
                "keepout mask 早于禁区文件，请在 Web 保存禁区后重新生成 mask："
                f"{zones_path}"
            )
    return []


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    map_yaml_path = LaunchConfiguration(
        'map',
        default='/home/sunrise/LD/maps/map_guosai_1.yaml'
    )
    nav2_param_path = LaunchConfiguration('params_file', default=os.path.join(
        get_package_share_directory('ldlidar'), 'config', 'nav2_params.yaml'))
    enable_keepout = LaunchConfiguration('enable_keepout')
    keepout_mask = LaunchConfiguration('keepout_mask')
    serial_port = LaunchConfiguration('serial_port')
    baudrate = LaunchConfiguration('baudrate')
    sensor_publish_rate = LaunchConfiguration('sensor_publish_rate')
    odom_frame = LaunchConfiguration('odom_frame')
    base_frame = LaunchConfiguration('base_frame')
    publish_tf = LaunchConfiguration('publish_tf')
    enable_cmd_vel = LaunchConfiguration('enable_cmd_vel')
    max_linear_x = LaunchConfiguration('max_linear_x')
    max_angular_z = LaunchConfiguration('max_angular_z')
    cmd_timeout = LaunchConfiguration('cmd_timeout')
    serial_command_rate = LaunchConfiguration('serial_command_rate')
    emergency_stop = LaunchConfiguration('emergency_stop')
    launch_rviz = LaunchConfiguration('launch_rviz')

    configured_nav2_params = RewrittenYaml(
        source_file=nav2_param_path,
        param_rewrites={
            (
                'global_costmap.global_costmap.ros__parameters.'
                'keepout_filter.enabled'
            ): enable_keepout,
            (
                'local_costmap.local_costmap.ros__parameters.'
                'keepout_filter.enabled'
            ): enable_keepout,
        },
        convert_types=True,
    )

    radar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('ldlidar'),
                'launch',
                'ld14p.launch.py',
            )
        ),
        launch_arguments={
            'serial_port': serial_port,
            'baudrate': baudrate,
            'sensor_publish_rate': sensor_publish_rate,
            'odom_frame': odom_frame,
            'base_frame': base_frame,
            'publish_tf': publish_tf,
            'enable_cmd_vel': enable_cmd_vel,
            'max_linear_x': max_linear_x,
            'max_angular_z': max_angular_z,
            'cmd_timeout': cmd_timeout,
            'serial_command_rate': serial_command_rate,
            'emergency_stop': emergency_stop,
        }.items(),
    )

    nav2_bringup_dir = get_package_share_directory('nav2_bringup')
    rviz_config_dir = os.path.join(
        nav2_bringup_dir,
        'rviz',
        'nav2_default_view.rviz',
    )

    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_bringup_dir, 'launch', 'bringup_launch.py')
        ),
        launch_arguments={
            'map': map_yaml_path,
            'use_sim_time': use_sim_time,
            'params_file': configured_nav2_params
        }.items(),
    )

    keepout_mask_server = Node(
        package='nav2_map_server',
        executable='map_server',
        name='keepout_filter_mask_server',
        output='screen',
        parameters=[
            configured_nav2_params,
            {
                'use_sim_time': use_sim_time,
                'yaml_filename': keepout_mask,
            },
        ],
        condition=IfCondition(enable_keepout),
    )

    keepout_filter_info_server = Node(
        package='nav2_map_server',
        executable='costmap_filter_info_server',
        name='keepout_costmap_filter_info_server',
        output='screen',
        parameters=[
            configured_nav2_params,
            {'use_sim_time': use_sim_time},
        ],
        condition=IfCondition(enable_keepout),
    )

    keepout_lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='keepout_lifecycle_manager',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'autostart': True,
            'node_names': [
                'keepout_filter_mask_server',
                'keepout_costmap_filter_info_server',
            ],
        }],
        condition=IfCondition(enable_keepout),
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_dir],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen',
        condition=IfCondition(launch_rviz),
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false',
                              description='Use simulation clock if true'),
        DeclareLaunchArgument('map', default_value=map_yaml_path,
                              description='Full path to map file to load'),
        DeclareLaunchArgument('params_file', default_value=nav2_param_path,
                              description='Full path to param file to load'),
        DeclareLaunchArgument('serial_port', default_value='/dev/ttyS2'),
        DeclareLaunchArgument('baudrate', default_value='115200'),
        DeclareLaunchArgument('sensor_publish_rate', default_value='10.0'),
        DeclareLaunchArgument('odom_frame', default_value='odom'),
        DeclareLaunchArgument('base_frame', default_value='base_footprint'),
        DeclareLaunchArgument('publish_tf', default_value='true'),
        DeclareLaunchArgument('enable_cmd_vel', default_value='true'),
        DeclareLaunchArgument('max_linear_x', default_value='0.08'),
        DeclareLaunchArgument('max_angular_z', default_value='0.40'),
        DeclareLaunchArgument('cmd_timeout', default_value='0.3'),
        DeclareLaunchArgument('serial_command_rate', default_value='10.0'),
        DeclareLaunchArgument('emergency_stop', default_value='true'),
        DeclareLaunchArgument(
            'launch_rviz',
            default_value='true',
            description='Start RViz for navigation visualization',
        ),
        DeclareLaunchArgument(
            'enable_keepout',
            default_value='false',
            description='Enable static Nav2 keepout filters',
        ),
        DeclareLaunchArgument(
            'keepout_mask',
            default_value='',
            description='Full path to the keepout mask YAML',
        ),

        OpaqueFunction(function=_validate_keepout_configuration),

        radar_launch,

        keepout_mask_server,
        keepout_filter_info_server,
        keepout_lifecycle_manager,
        nav2_launch,
        rviz_node,
    ])
