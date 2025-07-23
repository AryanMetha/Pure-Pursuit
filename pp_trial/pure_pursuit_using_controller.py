#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry, Path
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header, ColorRGBA
import math
import numpy as np
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

class PurePursuit(Node):
    def __init__(self):
        super().__init__('pure_pursuit_controller')
        
        # Parameters
        self.declare_parameter('lookahead_distance', 1.0)
        self.declare_parameter('linear_velocity', 0.8)
        self.declare_parameter('angular_velocity_limit', 1.0)
        self.declare_parameter('goal_tolerance', 0.2)
        
        
        self.lookahead_distance = self.get_parameter('lookahead_distance').value
        self.linear_velocity = self.get_parameter('linear_velocity').value
        self.angular_velocity_limit = self.get_parameter('angular_velocity_limit').value
        self.goal_tolerance = self.get_parameter('goal_tolerance').value
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Waypoints 
        self.path = [
            (0.0, 0.0),
            (2.0, 0.0),
            (4.0, 1.0),
            (6.0, 3.0),
            (8.0, 4.0),
            (10.0, 4.0),
            (12.0, 3.0),
            (14.0, 1.0),
            (16.0, 0.0),
            (18.0, -1.0),
            (20.0, -2.0),
            (22.0, -2.0),
            (24.0, -1.0),
            (26.0, 0.0),
            (28.0, 1.0),
            (30.0, 2.0)
        ]
        
        self.current_index = 0
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.path_completed = False
        
        # Publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.path_pub = self.create_publisher(Path, '/pure_pursuit_path', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/pure_pursuit_markers', 10)
        
        # Subscribers
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        
        self.viz_timer = self.create_timer(0.1, self.publish_visualization)
        
        # Publish the path 
        self.publish_path()
        
        self.get_logger().info("Pure Pursuit Controller Started")
        self.get_logger().info(f"Lookahead distance: {self.lookahead_distance}")
        self.get_logger().info(f"Linear velocity: {self.linear_velocity}")
        self.get_logger().info(f"Path has {len(self.path)} waypoints")

    def odom_callback(self, msg):
        # Get current robot pose
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        
        # Get yaw from quaternion
        orientation = msg.pose.pose.orientation
        self.robot_yaw = self.quaternion_to_yaw(orientation)
        
        # Check if we've completed the path
        if self.path_completed:
            self.cmd_pub.publish(Twist())
            return
        
        # Find goal point
        goal = self.find_lookahead_point(self.robot_x, self.robot_y)
        if goal is None:
            self.get_logger().info("Path completed!")
            self.path_completed = True
            self.cmd_pub.publish(Twist())
            return
        
        # Calculate control commands
        cmd = self.calculate_control_command(goal)
        self.cmd_pub.publish(cmd)

        # Broadcast odom → base_link TF
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'

        t.transform.translation.x = self.robot_x
        t.transform.translation.y = self.robot_y
        t.transform.translation.z = 0.0
        orientation = msg.pose.pose.orientation
        t.transform.rotation = orientation

        self.tf_broadcaster.sendTransform(t)


    def find_lookahead_point(self, x, y):
        """Find the lookahead point on the path"""
        min_dist_to_path = float('inf')
        closest_index = self.current_index
        
        # Find the closest point on the path
        for i in range(self.current_index, len(self.path)):
            px, py = self.path[i]
            dist = math.hypot(px - x, py - y)
            if dist < min_dist_to_path:
                min_dist_to_path = dist
                closest_index = i
        
        # Update current index to the closest point
        self.current_index = closest_index
        
        # Look for a point at lookahead distance
        for i in range(self.current_index, len(self.path)):
            px, py = self.path[i]
            dist = math.hypot(px - x, py - y)
            
            if dist >= self.lookahead_distance:
                return (px, py)
        
        # If we can't find a point at lookahead distance, check if we're close to the end
        last_point = self.path[-1]
        dist_to_end = math.hypot(last_point[0] - x, last_point[1] - y)
        
        if dist_to_end < self.goal_tolerance:
            return None  # We've reached the end
        else:
            return last_point  # Return the last point if we're close to the end

    def calculate_control_command(self, goal):
        """Calculate the control command using pure pursuit algorithm"""
        goal_x, goal_y = goal
        
        # Calculate relative position of goal
        dx = goal_x - self.robot_x
        dy = goal_y - self.robot_y
        
        # Transform goal to robot frame
        local_x = math.cos(-self.robot_yaw) * dx - math.sin(-self.robot_yaw) * dy
        local_y = math.sin(-self.robot_yaw) * dx + math.cos(-self.robot_yaw) * dy
        
        # Calculate distance to goal
        distance_to_goal = math.hypot(local_x, local_y)
        
        # Compute curvature using the pure pursuit formula
        if distance_to_goal < 0.01:  # Very close to goal
            curvature = 0.0
        else:
            curvature = 2 * local_y / (distance_to_goal ** 2)
        
        # Create control command
        cmd = Twist()
        cmd.linear.x = self.linear_velocity
        cmd.angular.z = curvature * self.linear_velocity
        
        # Limit angular velocity
        cmd.angular.z = max(-self.angular_velocity_limit, 
                           min(self.angular_velocity_limit, cmd.angular.z))
        
        return cmd

    def publish_path(self):
        """Publish the path for visualization in RViz"""
        path_msg = Path()
        path_msg.header.frame_id = "odom"
        path_msg.header.stamp = self.get_clock().now().to_msg()
        
        for point in self.path:
            pose = PoseStamped()
            pose.header.frame_id = "odom"
            pose.pose.position.x = float(point[0])
            pose.pose.position.y = float(point[1])
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        
        self.path_pub.publish(path_msg)

    def publish_visualization(self):
        """Publish visualization markers"""
        marker_array = MarkerArray()
        
        # Current waypoint marker
        if self.current_index < len(self.path):
            current_waypoint = Marker()
            current_waypoint.header.frame_id = "odom"
            current_waypoint.header.stamp = self.get_clock().now().to_msg()
            current_waypoint.ns = "current_waypoint"
            current_waypoint.id = 0
            current_waypoint.type = Marker.SPHERE
            current_waypoint.action = Marker.ADD
            current_waypoint.pose.position.x = float(self.path[self.current_index][0])
            current_waypoint.pose.position.y = float(self.path[self.current_index][1])
            current_waypoint.pose.position.z = 0.1
            current_waypoint.scale.x = 0.3
            current_waypoint.scale.y = 0.3
            current_waypoint.scale.z = 0.3
            current_waypoint.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)  # Red
            marker_array.markers.append(current_waypoint)
        
        # Lookahead circle
        lookahead_circle = Marker()
        lookahead_circle.header.frame_id = "odom"
        lookahead_circle.header.stamp = self.get_clock().now().to_msg()
        lookahead_circle.ns = "lookahead_circle"
        lookahead_circle.id = 1
        lookahead_circle.type = Marker.CYLINDER
        lookahead_circle.action = Marker.ADD
        lookahead_circle.pose.position.x = self.robot_x
        lookahead_circle.pose.position.y = self.robot_y
        lookahead_circle.pose.position.z = 0.0
        lookahead_circle.scale.x = self.lookahead_distance * 2
        lookahead_circle.scale.y = self.lookahead_distance * 2
        lookahead_circle.scale.z = 0.01
        lookahead_circle.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.3)  # Transparent green
        marker_array.markers.append(lookahead_circle)
        
        # Goal point marker
        goal = self.find_lookahead_point(self.robot_x, self.robot_y)
        if goal is not None:
            goal_marker = Marker()
            goal_marker.header.frame_id = "odom"
            goal_marker.header.stamp = self.get_clock().now().to_msg()
            goal_marker.ns = "goal_point"
            goal_marker.id = 2
            goal_marker.type = Marker.SPHERE
            goal_marker.action = Marker.ADD
            goal_marker.pose.position.x = float(goal[0])
            goal_marker.pose.position.y = float(goal[1])
            goal_marker.pose.position.z = 0.2
            goal_marker.scale.x = 0.2
            goal_marker.scale.y = 0.2
            goal_marker.scale.z = 0.2
            goal_marker.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0)  # Blue
            marker_array.markers.append(goal_marker)
        
        self.marker_pub.publish(marker_array)

    def quaternion_to_yaw(self, q):
        """Convert quaternion to yaw angle"""
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

def main(args=None):
    rclpy.init(args=args)
    node = PurePursuit()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()