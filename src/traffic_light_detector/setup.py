import os
from glob import glob

from setuptools import setup


package_name = "traffic_light_detector"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="xycar",
    maintainer_email="hackathontensor@gmail.com",
    description="OpenCV four-aspect traffic-light detection with halo and structure validation",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "traffic_light_detector_node = traffic_light_detector.traffic_light_detector_node:main",
            "hsv_tuner_node = traffic_light_detector.hsv_tuner_node:main",
        ],
    },
)
