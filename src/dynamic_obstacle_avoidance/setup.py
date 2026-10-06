import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'dynamic_obstacle_avoidance'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='xytron',
    maintainer_email='xytron@todo.todo',
    description='Rule-based LiDAR centerline obstacle avoidance with E2E handoff',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'dynamic_obstacle_avoidance = dynamic_obstacle_avoidance.dynamic_obstacle_avoidance:main',
            'dynamic_obstacle_log = dynamic_obstacle_avoidance.dynamic_obstacle_log:main',
        ],
    },
)
