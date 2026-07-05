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
            'bsmc_circle = amr_control.bsmc_circle:main',
            'robot_serial_bridge = amr_control.robot_serial_bridge:main',
            'camera_node = amr_control.camera:main',
            'keyboard_teleop = amr_control.keyboard_teleop:main',
            'odom_logger = amr_control.odom_logger:main',
            'debug_yaw = amr_control.debug_yaw:main',
            'homography_calibration = amr_control.homography_calibration:main',
        ],
    },
)
