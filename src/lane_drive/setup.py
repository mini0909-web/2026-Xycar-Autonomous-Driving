from glob import glob
import os

from setuptools import find_packages, setup


package_name = 'lane_drive'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='xytron',
    maintainer_email='xytron@todo.todo',
    description='ROS 2 camera lane detection and PD steering for Xycar',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'hsv_tuner_node = lane_drive.hsv_tuner_node:main',
            'lane_drive_node = lane_drive.lane_drive_node:main',
        ],
    },
)
