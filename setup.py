from glob import glob

from setuptools import find_packages, setup


package_name = "mpose_live"

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
    description="MegaPose UDP bridge and live Vicon pose comparison",
    license="TODO: License declaration",
    extras_require={"test": ["pytest"]},
    entry_points={
        "console_scripts": [
            "mpose_bridge_node = mpose_live.mpose_bridge_node:main",
            "gt_error_node = mpose_live.gt_error_node:main",
        ],
    },
)