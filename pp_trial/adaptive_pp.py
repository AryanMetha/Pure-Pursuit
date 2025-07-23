#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry, Path
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header, ColorRGBA
import math
import numpy as np
import csv
import os
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped


class AdaptiveLookaheadPurePursuit(Node):
    def __init__(self):
        super().__init__('adaptive_lookahead_pure_pursuit_controller')
        
        # Parameters
        self.declare_parameter('base_linear_velocity', 0.8)
        self.declare_parameter('angular_velocity_limit', 1.0)
        self.declare_parameter('goal_tolerance', 0.2)
        self.declare_parameter('control_frequency', 50.0)
        self.declare_parameter('beta', 0.7)  # Weight for exit speed vs area minimization
        self.declare_parameter('simulation_time', 3.0)  # Time to simulate each lookahead
        self.declare_parameter('csv_output_path', 'adaptive_lookahead_results.csv')
        
        self.base_linear_velocity = self.get_parameter('base_linear_velocity').value
        self.angular_velocity_limit = self.get_parameter('angular_velocity_limit').value
        self.goal_tolerance = self.get_parameter('goal_tolerance').value
        self.control_frequency = self.get_parameter('control_frequency').value
        self.beta = self.get_parameter('beta').value
        self.simulation_time = self.get_parameter('simulation_time').value
        self.csv_output_path = self.get_parameter('csv_output_path').value
        
        # Lookahead distances to test
        self.test_lookaheads = [ 0.5,0.6, 0.7,0.8,0.9,1.0,1.3,1.5]
        
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Robot state
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.robot_linear_vel = 0.0
        self.robot_angular_vel = 0.0
        self.delta = 0.0  # Deviation from the path
        
        # Path definition (same as reference)
        self.path = [
            # Bottom straight
            (0.0, 0.0), (2.0, 0.0), (4.0, 1.0), (6.0, 0.0),
            # Right curve upward
            (7.0, 0.5), (7.5, 1.5), (8.0, 3.0),
            # Top straight (left)
            (6.0, 4.0), (4.0, 3.0), (2.0, 4.0), (0.0, 4.0),
            # Left curve downward
            (-1.0, 3.0), (-1.5, 2.0), (-1.0, 1.0), (0.0, 0.0)
        ]

        self.current_index = 0
        self.optimal_lookaheads = {}  # Store optimal lookahead for each waypoint
        self.simulation_results = []  # Store all simulation results
        
        # Current simulation state
        self.current_lookahead = self.test_lookaheads[0]
        self.current_waypoint_test = 0
        self.current_lookahead_test = 0
        self.is_testing = True
        self.test_start_time = None
        self.test_trajectory = []  # Store trajectory during test
        
        # Publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.path_pub = self.create_publisher(Path, '/adaptive_pp_path', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/adaptive_pp_markers', 10)
        
        # Timers
        self.control_timer = self.create_timer(1.0 / self.control_frequency, self.control_loop)
        self.viz_timer = self.create_timer(0.1, self.publish_visualization)
        self.odom_timer = self.create_timer(1.0 / self.control_frequency, self.publish_odometry)
        
        # Time tracking
        self.last_time = self.get_clock().now()
        
        # Initialize CSV file
        self.initialize_csv()
        
        # Publish the path
        self.publish_path()
        
        # Start testing
        self.start_waypoint_test()
        
        self.get_logger().info("Adaptive Lookahead Pure Pursuit Controller Started")
        self.get_logger().info(f"Testing lookahead distances: {self.test_lookaheads}")
        self.get_logger().info(f"Beta parameter: {self.beta}")
        self.get_logger().info(f"Path has {len(self.path)} waypoints")

    def initialize_csv(self):
        """Initialize the CSV file with headers"""
        headers = ['waypoint_index', 'waypoint_x', 'waypoint_y', 'lookahead_distance', 
                   'exit_speed', 'area_delta', 'combined_score', 'is_optimal']
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.csv_output_path) if os.path.dirname(self.csv_output_path) else '.', exist_ok=True)
        
        with open(self.csv_output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)

    def start_waypoint_test(self):
        """Initialize test for current waypoint"""
        if self.current_waypoint_test >= len(self.path):
            self.get_logger().info("All waypoints tested! Results saved to CSV.")
            self.is_testing = False
            return
        
        waypoint = self.path[self.current_waypoint_test]
        self.get_logger().info(f"Testing waypoint {self.current_waypoint_test}: {waypoint}")
        
        # Reset for new waypoint test
        self.current_lookahead_test = 0
        self.start_lookahead_test()

    def start_lookahead_test(self):
        """Initialize test for current lookahead distance"""
        if self.current_lookahead_test >= len(self.test_lookaheads):
            # Finished testing all lookaheads for this waypoint
            self.analyze_waypoint_results()
            self.current_waypoint_test += 1
            self.start_waypoint_test()
            return
        
        waypoint = self.path[self.current_waypoint_test]
        self.current_lookahead = self.test_lookaheads[self.current_lookahead_test]
        
        self.get_logger().info(f"Testing lookahead {self.current_lookahead} at waypoint {self.current_waypoint_test}")
        
        # Set robot position at waypoint
        self.robot_x = waypoint[0]
        self.robot_y = waypoint[1]
        
        
        # Set orientation along previous path segment
        self.set_robot_orientation()
        
        # Reset velocities
        self.robot_linear_vel = self.base_linear_velocity
        self.robot_angular_vel = 0.0
        
        # Initialize test tracking
        self.test_start_time = self.get_clock().now()
        self.test_trajectory = [(self.robot_x, self.robot_y)]
        
        # Set current index for path following
        self.current_index = self.current_waypoint_test

    def set_robot_orientation(self):
        """Set robot orientation along the previous path segment"""
        if self.current_waypoint_test == 0:
            # First waypoint, look towards next waypoint
            if len(self.path) > 1:
                dx = self.path[1][0] - self.path[0][0]
                dy = self.path[1][1] - self.path[0][1]
                self.robot_yaw = math.atan2(dy, dx)
            else:
                self.robot_yaw = 0.0
        else:
            # Look along direction from previous waypoint
            prev_wp = self.path[self.current_waypoint_test - 1]
            curr_wp = self.path[self.current_waypoint_test]
            dx = curr_wp[0] - prev_wp[0]
            dy = curr_wp[1] - prev_wp[1]
            self.robot_yaw = math.atan2(dy, dx)

    def control_loop(self):
        """Main control loop"""
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        self.last_time = current_time
        
        if not self.is_testing:
            # Stop the robot
            cmd = Twist()
            self.cmd_pub.publish(cmd)
            return
        
        # Check if simulation time exceeded
        if self.test_start_time is not None:
            elapsed_time = (current_time - self.test_start_time).nanoseconds / 1e9
            if elapsed_time >= self.simulation_time:
                self.finish_lookahead_test()
                return
        
        # Update robot pose
        self.update_robot_pose(dt)
        
        # Record trajectory
        if len(self.test_trajectory) == 0 or \
           math.hypot(self.robot_x - self.test_trajectory[-1][0], 
                     self.robot_y - self.test_trajectory[-1][1]) > 0.01:
            self.test_trajectory.append((self.robot_x, self.robot_y))
        
        # Find goal point
        goal = self.find_lookahead_point(self.robot_x, self.robot_y)
        if goal is None:
            self.finish_lookahead_test()
            return
        
        # Calculate and publish control command
        cmd = self.calculate_control_command(goal)
        self.cmd_pub.publish(cmd)

    def finish_lookahead_test(self):
        """Finish current lookahead test and record results"""
        # Calculate exit speed (current speed)
        exit_speed = math.hypot(self.robot_linear_vel, 0.0)  # Only linear velocity
        
        # Calculate area delta between actual trajectory and reference segment
        area_delta = self.delta
        self.delta = 0.0  # Reset delta for next test
        
        # Store results
        waypoint = self.path[self.current_waypoint_test]
        result = {
            'waypoint_index': self.current_waypoint_test,
            'waypoint_x': waypoint[0],
            'waypoint_y': waypoint[1],
            'lookahead_distance': self.current_lookahead,
            'exit_speed': exit_speed, #useless rn
            'area_delta': area_delta,
            'trajectory': self.test_trajectory.copy()
        }
        self.simulation_results.append(result)
        
        self.get_logger().info(f"Lookahead {self.current_lookahead}: exit_speed={exit_speed:.3f}, area_delta={area_delta:.6f}")
        
        # Move to next lookahead test
        self.current_lookahead_test += 1
        self.start_lookahead_test()

    def calculate_area_delta(self):
        """Calculate area between actual trajectory and reference path segment"""
        if len(self.test_trajectory) < 2:
            return 0.0
        
        # Get reference segment
        waypoint_idx = self.current_waypoint_test
        if waypoint_idx == 0:
            # Use segment to next waypoint
            if len(self.path) > 1:
                ref_start = self.path[0]
                ref_end = self.path[1]
            else:
                return 0.0
        else:
            # Use segment from previous waypoint
            ref_start = self.path[waypoint_idx - 1]
            ref_end = self.path[waypoint_idx]
        
        # Calculate area using the shoelace formula for the polygon formed by
        # actual trajectory and reference segment
        try:
            # Create closed polygon: trajectory + reference line back
            polygon_points = self.test_trajectory.copy()
            
            # Add reference line points to close the polygon
            # Project trajectory endpoints onto reference line
            start_proj = self.project_point_on_line(self.test_trajectory[0], ref_start, ref_end)
            end_proj = self.project_point_on_line(self.test_trajectory[-1], ref_start, ref_end)
            
            # Close the polygon
            polygon_points.append(end_proj)
            polygon_points.append(start_proj)
            
            # Calculate area using shoelace formula
            area = 0.0
            n = len(polygon_points)
            for i in range(n):
                j = (i + 1) % n
                area += polygon_points[i][0] * polygon_points[j][1]
                area -= polygon_points[j][0] * polygon_points[i][1]
            
            return abs(area) / 2.0
            
        except Exception as e:
            self.get_logger().warn(f"Error calculating area delta: {e}")
            return 0.0

    def project_point_on_line(self, point, line_start, line_end):
        """Project a point onto a line segment"""
        px, py = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        # Vector from line_start to line_end
        dx = x2 - x1
        dy = y2 - y1
        
        if dx == 0 and dy == 0:
            return line_start
        
        # Parameter t for projection
        t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
        t = max(0, min(1, t))  # Clamp to line segment
        
        # Projected point
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        
        return (proj_x, proj_y)

    def analyze_waypoint_results(self):
        """Analyze results for current waypoint and find optimal lookahead"""
        waypoint_results = [r for r in self.simulation_results 
                           if r['waypoint_index'] == self.current_waypoint_test]
        
        if not waypoint_results:
            return
        
        best_lookahead = self.test_lookaheads[np.argmin([r['area_delta'] for r in waypoint_results])]
        
        
        waypoint = self.path[self.current_waypoint_test]
        self.get_logger().info(f"Optimal lookahead for waypoint {self.current_waypoint_test} {waypoint}: {best_lookahead}")
        self.write_waypoint_results_to_csv( waypoint_results, best_lookahead)
        print("Wrote results to CSV")

    def write_waypoint_results_to_csv(self, waypoint_results, best_lookahead):
        """Write waypoint results to CSV file"""
        with open(self.csv_output_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            self.get_logger().info(f"Writing results to {self.csv_output_path}")
            for result in waypoint_results:
                is_optimal = result['lookahead_distance'] == best_lookahead
                row = [
                    result['waypoint_index'],
                    result['waypoint_x'],
                    result['waypoint_y'],
                    result['lookahead_distance'],
                    result['exit_speed'],
                    result['area_delta'],
                    is_optimal
                ]
                writer.writerow(row)

    def update_robot_pose(self, dt):
        """Update robot pose based on current velocities"""
        if dt <= 0:
            return
        
        # Differential drive kinematics
        self.robot_x += self.robot_linear_vel * math.cos(self.robot_yaw) * dt
        self.robot_y += self.robot_linear_vel * math.sin(self.robot_yaw) * dt
        self.robot_yaw += self.robot_angular_vel * dt

        #delta deviation
        length = math.hypot(self.robot_x-self.path[self.current_waypoint_test][0], self.robot_y-self.path[self.current_waypoint_test][1])
        self.delta+=dt* abs(self.robot_angular_vel)*length
        
        # Normalize yaw
        self.robot_yaw = math.atan2(math.sin(self.robot_yaw), math.cos(self.robot_yaw))

    def find_lookahead_point(self, x, y):
        """Find the lookahead point on the path"""
        # Find closest point starting from current index
        min_dist_to_path = float('inf')
        closest_index = self.current_index
        
        for i in range(self.current_index, len(self.path)):
            px, py = self.path[i]
            dist = math.hypot(px - x, py - y)
            if dist < min_dist_to_path:
                min_dist_to_path = dist
                closest_index = i
        
        self.current_index = closest_index
        
        # Look for point at lookahead distance
        for i in range(self.current_index, len(self.path)):
            px, py = self.path[i]
            dist = math.hypot(px - x, py - y)
            
            if dist >= self.current_lookahead:
                return (px, py)
        
        # Check if close to end
        last_point = self.path[-1]
        dist_to_end = math.hypot(last_point[0] - x, last_point[1] - y)
        
        if dist_to_end < self.goal_tolerance:
            return None
        else:
            return last_point

    def calculate_control_command(self, goal):
        """Calculate control command using pure pursuit"""
        goal_x, goal_y = goal
        
        # Calculate relative position
        dx = goal_x - self.robot_x
        dy = goal_y - self.robot_y
        
        # Transform to robot frame
        local_x = math.cos(-self.robot_yaw) * dx - math.sin(-self.robot_yaw) * dy
        local_y = math.sin(-self.robot_yaw) * dx + math.cos(-self.robot_yaw) * dy
        
        # Calculate distance and curvature
        distance_to_goal = math.hypot(local_x, local_y)
        
        if distance_to_goal < 0.01:
            curvature = 0.0
        else:
            curvature = 2 * local_y / (distance_to_goal ** 2)
        
        # Create command
        cmd = Twist()
        cmd.linear.x = self.base_linear_velocity
        cmd.angular.z = curvature * self.base_linear_velocity
        
        # Limit angular velocity
        cmd.angular.z = np.clip(cmd.angular.z, -self.angular_velocity_limit, self.angular_velocity_limit)
        
        # Update robot velocities for simulation
        self.robot_linear_vel = cmd.linear.x
        self.robot_angular_vel = cmd.angular.z
        
        return cmd

    def publish_odometry(self):
        """Publish odometry message"""
        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = 'base_link'
        
        # Position
        odom_msg.pose.pose.position.x = self.robot_x
        odom_msg.pose.pose.position.y = self.robot_y
        odom_msg.pose.pose.position.z = 0.0
        
        # Orientation
        quat = self.yaw_to_quaternion(self.robot_yaw)
        odom_msg.pose.pose.orientation.x = quat[0]
        odom_msg.pose.pose.orientation.y = quat[1]
        odom_msg.pose.pose.orientation.z = quat[2]
        odom_msg.pose.pose.orientation.w = quat[3]
        
        # Velocity
        odom_msg.twist.twist.linear.x = self.robot_linear_vel
        odom_msg.twist.twist.angular.z = self.robot_angular_vel
        
        # Covariance
        for i in range(36):
            if i in [0, 7, 14]:
                odom_msg.pose.covariance[i] = 0.1
                odom_msg.twist.covariance[i] = 0.1
            elif i in [21, 28, 35]:
                odom_msg.twist.covariance[i] = 0.05
                odom_msg.pose.covariance[i] = 0.05
        
        self.odom_pub.publish(odom_msg)
        self.broadcast_tf()

    def broadcast_tf(self):
        """Broadcast transform from odom to base_link"""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'
        
        t.transform.translation.x = self.robot_x
        t.transform.translation.y = self.robot_y
        t.transform.translation.z = 0.0
        
        quat = self.yaw_to_quaternion(self.robot_yaw)
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]
        
        self.tf_broadcaster.sendTransform(t)

    def publish_path(self):
        """Publish the path for visualization"""
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
        
        # Current test waypoint
        if self.is_testing and self.current_waypoint_test < len(self.path):
            waypoint = self.path[self.current_waypoint_test]
            test_marker = Marker()
            test_marker.header.frame_id = "odom"
            test_marker.header.stamp = self.get_clock().now().to_msg()
            test_marker.ns = "test_waypoint"
            test_marker.id = 0
            test_marker.type = Marker.SPHERE
            test_marker.action = Marker.ADD
            test_marker.pose.position.x = float(waypoint[0])
            test_marker.pose.position.y = float(waypoint[1])
            test_marker.pose.position.z = 0.1
            test_marker.scale.x = 0.4
            test_marker.scale.y = 0.4
            test_marker.scale.z = 0.4
            test_marker.color = ColorRGBA(r=1.0, g=0.5, b=0.0, a=1.0)  # Orange
            marker_array.markers.append(test_marker)
        
        # Current lookahead circle
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
        lookahead_circle.scale.x = self.current_lookahead * 2
        lookahead_circle.scale.y = self.current_lookahead * 2
        lookahead_circle.scale.z = 0.01
        lookahead_circle.color = ColorRGBA(r=0.0, g=1.0, b=1.0, a=0.3)  # Transparent cyan
        marker_array.markers.append(lookahead_circle)
        
        # Robot marker
        robot_marker = Marker()
        robot_marker.header.frame_id = "odom"
        robot_marker.header.stamp = self.get_clock().now().to_msg()
        robot_marker.ns = "robot_position"
        robot_marker.id = 2
        robot_marker.type = Marker.ARROW
        robot_marker.action = Marker.ADD
        robot_marker.pose.position.x = self.robot_x
        robot_marker.pose.position.y = self.robot_y
        robot_marker.pose.position.z = 0.1
        
        quat = self.yaw_to_quaternion(self.robot_yaw)
        robot_marker.pose.orientation.x = quat[0]
        robot_marker.pose.orientation.y = quat[1]
        robot_marker.pose.orientation.z = quat[2]
        robot_marker.pose.orientation.w = quat[3]
        
        robot_marker.scale.x = 0.3
        robot_marker.scale.y = 0.05
        robot_marker.scale.z = 0.05
        robot_marker.color = ColorRGBA(r=1.0, g=0.0, b=1.0, a=1.0)  # Magenta
        marker_array.markers.append(robot_marker)
        
        self.marker_pub.publish(marker_array)

    def yaw_to_quaternion(self, yaw):
        """Convert yaw angle to quaternion"""
        half_yaw = yaw * 0.5
        return [
            0.0,
            0.0,
            math.sin(half_yaw),
            math.cos(half_yaw)
        ]

def main(args=None):
    rclpy.init(args=args)
    node = AdaptiveLookaheadPurePursuit()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()