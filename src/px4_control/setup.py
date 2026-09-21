from setuptools import setup

package_name = 'px4_control'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    entry_points={
        'console_scripts': [
            'takeoff_land = px4_control.takeoff_land:main',
            'offboard_node = px4_control.offboard_node:main',
        ],
    },
)
