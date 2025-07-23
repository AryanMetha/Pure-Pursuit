from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

import os

def generate_launch_description():
    pkg_share = get_package_share_directory('pp_trial')
    
    # URDF file
    urdf_file = os.path.join(pkg_share, 'urdf', 'diffbot2.urdf')
    
    # RViz configuration file
    rviz_config = os.path.join(pkg_share, 'config', 'diffbot.rviz')

    return LaunchDescription([
        # Robot State Publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': open(
                os.path.join(
                    get_package_share_directory('pp_trial'),
                    'urdf/diffbot2.urdf'
                ), 'r').read()}]
        ),

        # Joint State Broadcaster
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=['joint_state_broadcaster', '--controller-manager-timeout', '60'],
            output='screen',
        ),

        # Diff Drive Controller
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=['diff_drive_controller', '--controller-manager-timeout', '60'],
            output='screen',
        )
    ])
