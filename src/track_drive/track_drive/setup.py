from setuptools import find_packages, setup
import os
from glob import glob

package_name = "track_drive"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
            ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"),
            glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="a",
    maintainer_email="a@todo.todo",
    description="Xycar autonomous driving package",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "traffic_light = track_drive.traffic_light:main",
            "e2e_drive     = track_drive.e2e_drive:main",
            "e2e_pure      = track_drive.e2e_pure:main",
            "e2e_pure_clahe = track_drive.e2e_pure_clahe:main",
            "data_logger   = track_drive.data_logger:main",
            "teleop_key    = track_drive.teleop_key:main",
            "drive_mux     = track_drive.mux_node:main",
            "e2e_debug     = track_drive.e2e_debug:main",
        ],
    },
)
