from setuptools import find_packages, setup
from glob import glob

package_name = "mpose_consumer"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="abhi",
    maintainer_email="coderadonis@gmail.com",
    description="live Vicon pose comparison and controller",
    license="TODO: License declaration",
    entry_points={
        "console_scripts": [
            f"gt_error_node = {package_name}.gt_error_node:main",
            f"controller_node = {package_name}.controller_node:main",
            f"rel_pose_node = {package_name}.rel_pose_node:main"
        ],
    },
)
