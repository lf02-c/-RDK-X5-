#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration

'''
Parameter Description:
---
- Set laser scan directon: 
  1. Set counterclockwise, example: {'laser_scan_dir': True}
  2. Set clockwise,        example: {'laser_scan_dir': False}
- Angle crop setting, Mask data within the set angle range:
  1. Enable angle crop fuction:
    1.1. enable angle crop,  example: {'enable_angle_crop_func': True}
    1.2. disable angle crop, example: {'enable_angle_crop_func': False}
  2. Angle cropping interval setting:
  - The distance and intensity data within the set angle range will be set to 0.
  - angle >= 'angle_crop_min' and angle <= 'angle_crop_max' which is [angle_crop_min, angle_crop_max], unit is degress.
    example:
      {'angle_crop_min': 135.0}
      {'angle_crop_max': 225.0}
      which is [135.0, 225.0], angle unit is degress.
'''

def generate_launch_description():
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

  # LDROBOT LiDAR publisher node
  ldlidar_node = Node(
      package='ldlidar',
      executable='ldlidar',
      name='ldlidar_publisher_ld14p',
      output='screen',
      parameters=[
        {'product_name': 'LDLiDAR_LD14P'},
        {'topic_name': 'scan'},
        {'port_name': '/dev/ttyACM0'},
        {'frame_id': 'base_laser'},
        {'laser_scan_dir': True},
        {'enable_angle_crop_func': False},#单角度裁剪开关：值为False时表示不使用多角度裁剪，默认为False
        {'angle_crop_min': 135.0},#单角度裁剪开始值
        {'angle_crop_max': 225.0},#单角度裁剪结束值
        {'truncated_mode_': 0}#值为1表示使用多角度裁剪，同时enable_angle_crop_func设为False，角度值在/main.cpp中修改
      ]
  )

  # base_link to base_laser tf node
  base_link_to_laser_tf_node = Node(
    package='tf2_ros',
    executable='static_transform_publisher',
    name='base_link_to_base_laser_ld14p',
    arguments=['-0.05','0','0.25','0','0','0','base_footprint','base_laser']
  )

  scan_fre_node = ExecuteProcess(
    cmd=['ros2','run','ldlidar','LD14P_scan_fre.py']
  )

  odom_node = Node(
    package='autopatrol_robot',
    executable='stm32_bridge_node',
    name='stm32_bridge_node',
    output='screen',
    parameters=[
      {'serial_port': serial_port},
      {'baudrate': ParameterValue(baudrate, value_type=int)},
      {'sensor_publish_rate': ParameterValue(sensor_publish_rate, value_type=float)},
      {'odom_frame': odom_frame},
      {'base_frame': base_frame},
      {'publish_tf': ParameterValue(publish_tf, value_type=bool)},
      {'enable_cmd_vel': ParameterValue(enable_cmd_vel, value_type=bool)},
      {'max_linear_x': ParameterValue(max_linear_x, value_type=float)},
      {'max_angular_z': ParameterValue(max_angular_z, value_type=float)},
      {'cmd_timeout': ParameterValue(cmd_timeout, value_type=float)},
      {'serial_command_rate': ParameterValue(serial_command_rate, value_type=float)},
      {'emergency_stop': ParameterValue(emergency_stop, value_type=bool)}
    ]
  )



  # Define LaunchDescription variable
  ld = LaunchDescription()

  ld.add_action(DeclareLaunchArgument('serial_port', default_value='/dev/ttyS2'))
  ld.add_action(DeclareLaunchArgument('baudrate', default_value='115200'))
  ld.add_action(DeclareLaunchArgument('sensor_publish_rate', default_value='10.0'))
  ld.add_action(DeclareLaunchArgument('odom_frame', default_value='odom'))
  ld.add_action(DeclareLaunchArgument('base_frame', default_value='base_footprint'))
  ld.add_action(DeclareLaunchArgument('publish_tf', default_value='true'))
  ld.add_action(DeclareLaunchArgument('enable_cmd_vel', default_value='false'))
  ld.add_action(DeclareLaunchArgument('max_linear_x', default_value='0.08'))
  ld.add_action(DeclareLaunchArgument('max_angular_z', default_value='0.40'))
  ld.add_action(DeclareLaunchArgument('cmd_timeout', default_value='0.3'))
  ld.add_action(DeclareLaunchArgument('serial_command_rate', default_value='10.0'))
  ld.add_action(DeclareLaunchArgument('emergency_stop', default_value='true'))

  #ld.add_action(scan_fre_node)  #<!--调节雷达扫描频率，scan_fre扫描频率与雷达串口号请在LD14P_scan_fre.py文件中修改-->
  ld.add_action(base_link_to_laser_tf_node)
  ld.add_action(ldlidar_node)
  ld.add_action(scan_fre_node)
  ld.add_action(odom_node)

  

  return ld
