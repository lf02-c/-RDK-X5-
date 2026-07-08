from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
import rclpy
from rclpy.duration import Duration


def main():
    rclpy.init()
    navigator = BasicNavigator()
    # Wait for Nav2 to become active
    navigator.waitUntilNav2Active()
    # Set goal pose
    goal_pose = PoseStamped()
    goal_pose.header.frame_id = 'map'
    goal_pose.header.stamp = navigator.get_clock().now().to_msg()
    goal_pose.pose.position.x = 1.0
    goal_pose.pose.position.y =0.0
    goal_pose.pose.orientation.w = 1.0
    # Send goal and monitor progress
    navigator.goToPose(goal_pose)
    while not navigator.isTaskComplete():
        feedback = navigator.getFeedback()
        remaining = Duration.from_msg(feedback.estimated_time_remaining).nanoseconds / 1e9
        navigator.get_logger().info(f'Estimated time remaining: {remaining:.1f} s')
        # Cancel if navigation takes longer than 600 seconds
        if Duration.from_msg(feedback.navigation_time) > Duration(seconds=600.0):
            navigator.cancelTask()
    # Check final result
    result = navigator.getResult()
    if result == TaskResult.SUCCEEDED:
        navigator.get_logger().info('Navigation result: SUCCEEDED')
    elif result == TaskResult.CANCELED:
        navigator.get_logger().warn('Navigation result: CANCELED')
    elif result == TaskResult.FAILED:
        navigator.get_logger().error('Navigation result: FAILED')
    else:
        navigator.get_logger().error('Navigation result: UNKNOWN')

if __name__ == '__main__':
    main()