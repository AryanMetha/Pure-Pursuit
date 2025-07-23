import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    pkg_share = get_package_share_directory('pp_trial')
    
    # Gazebo launch
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')]),
    )
    
    # URDF file
    urdf_file = os.path.join(pkg_share, 'urdf', 'manual.urdf')
    rviz_config = os.path.join(pkg_share, 'config', 'diffbot.rviz')
    
    with open(urdf_file, 'r') as infp:
        robot_desc = infp.read()
    
    # Robot State Publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_desc,
            'use_sim_time': True
        }]
    )
    

    # Spawn robot in Gazebo
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-topic', 'robot_description',
                  '-entity', 'diffbot'],
        output='screen'
    )
    
    # Pure Pursuit Controller Node
    pure_pursuit_node = Node(
        package='pp_trial',
        executable='adaptive_pure_pursuit_node',
        name='pure_pursuit_controller',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'lookahead_distance': 1.5,
            'linear_velocity': 0.8,
            'angular_velocity_limit': 1.0,
            'goal_tolerance': 0.3
        }]

    )
    
    # RViz with visualization
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': True}],
        output='screen'
    )
    
    # Add a delay to start the pure pursuit node after the robot is spawned
    delayed_pure_pursuit = TimerAction(
        period=3.0,
        actions=[pure_pursuit_node]
    )
    
    return LaunchDescription([
        gazebo,
        robot_state_publisher,
        spawn_entity,
        delayed_pure_pursuit,
        rviz,
        
    ])