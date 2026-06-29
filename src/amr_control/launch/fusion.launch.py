from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    custom_ekf_params = PathJoinSubstitution([
        FindPackageShare('amr_control'),
        'config',
        'custom_ekf.yaml',
    ])

    return LaunchDescription([
        Node(
            package='amr_control',
            executable='camera_node',
            name='pose_estimation_publisher',
            output='screen',
            parameters=[{
                'prefilter_pose': False,
            }],
        ),
        Node(
            package='amr_control',
            executable='state_bridge',
            name='state_bridge_node',
            output='screen',
        ),
        Node(
            package='amr_control',
            executable='custom_ekf_node',
            name='custom_ekf_node',
            output='screen',
            parameters=[custom_ekf_params],
        ),
    ])
