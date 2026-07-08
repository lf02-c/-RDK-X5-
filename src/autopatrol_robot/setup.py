from setuptools import find_packages, setup

package_name = 'autopatrol_robot'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name+"/config", ['config/patrol_config.yaml']),
        ('share/' + package_name+"/config", ['config/slam_with_keyboard.rviz']),
        ('share/' + package_name + "/web", ['../web/index.html']),
        ('share/' + package_name + "/data", ['data/alarm_logs.json']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sunrise',
    maintainer_email='sunrise@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'patrol_node=autopatrol_robot.patrol_node:main',
            'door_detect_node=autopatrol_robot.door_detect_node:main',
            'door_detect_node_perf = autopatrol_robot.door_detect_node_perf:main',
            'door_detect_node_debug_compare = autopatrol_robot.door_detect_node_debug_compare:main',
            'alarm_node=autopatrol_robot.alarm_node:main',
            'robot_gui=autopatrol_robot.robot_gui:main',
            'ze07_node=autopatrol_robot.ze07_node:main',
            'web_bridge_node=autopatrol_robot.web_bridge_node:main',
            'sensor_sim_node=autopatrol_robot.sensor_sim_node:main',
            'serial_sensor_node=autopatrol_robot.serial_sensor_node:main',
            'stm32_bridge_node=autopatrol_robot.stm32_bridge_node:main',
            'lcd_node=autopatrol_robot.lcd_node:main',
       ],
    },
)
