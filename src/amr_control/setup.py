import os
from setuptools import setup
from glob import glob

package_name = 'amr_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools', 'pyserial'],
    zip_safe=True,
    maintainer='Admin',
    maintainer_email='admin@todo.todo',
    description='ROS2 package for AMR Control with Backstepping + SMC',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'state_bridge = amr_control.state_bridge:main',
            'custom_ekf_node = amr_control.custom_ekf_node:main',
            'bsmc_circle = amr_control.bsmc_paper_runner:circle_main',
            'bsmc_eight = amr_control.bsmc_paper_runner:eight_main',
            'bsmc_square = amr_control.bsmc_paper_runner:square_main',
            'backstepping_circle = amr_control.backstepping_circle:main',
            'backstepping_eight = amr_control.backstepping_eight:main',
            'backstepping_square = amr_control.backstepping_square:main',
            'espnow_paper_test = amr_control.espnow_paper_test:main',
            'robot_serial_bridge = amr_control.robot_serial_bridge:main',
            'camera_node = amr_control.camera:main',
            'camera_circle_square = amr_control.camera_profiles:circle_square_main',
            'camera_eight = amr_control.camera_profiles:eight_main',
            'keyboard_teleop = amr_control.keyboard_teleop:main',
            'odom_logger = amr_control.odom_logger:main',
            'paper_dashboard = amr_control.paper_dashboard:main',
            'paper_data_logger = amr_control.paper_data_logger:main',
            'paper_link_logger = amr_control.paper_link_logger:main',
            'paper_metrics = amr_control.paper_metrics:main',
            'paper_bsmc_sim = amr_control.paper_bsmc_sim:main',
            'bsmc_experiment = amr_control.bsmc_experiment:main',
            'debug_yaw = amr_control.debug_yaw:main',
            'homography_calibration = amr_control.homography_calibration:main',
            'optuna_bsmc_tuning = amr_control.optuna_bsmc_tuning:main',
            'spin_test = amr_control.spin_test:main',
            'square_tuning_report = amr_control.square_tuning_report:main',
            'eight_tuning_report = amr_control.eight_tuning_report:main',
            'circle_tuning_report = amr_control.circle_tuning_report:main',
        ],
    },
)
