from setuptools import find_packages, setup

package_name = 'px4_control'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kiwi',
    maintainer_email='lehuo10@rowan.edu',
    description='Control package for PX4',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'arm_disarm = px4_control.arm_disarm:main',
            'takeoff_land_mission_offboard = px4_control.takeoff_land_mission_offboard:main',
            'takeoff_land_mission_gps = px4_control.takeoff_land_mission_gps:main',
            'takeoff_hold_mission_offboard = px4_control.takeoff_hold_mission_offboard:main',
            'takeoff_hold_mission_gps = px4_control.takeoff_hold_mission_gps:main',
            'takeoff_land_takephoto_mission = px4_control.takeoff_land_takephoto_mission:main',
            'record_drone_state   = px4_control.record_drone_state:main',
            'fly_setpoints_mission = px4_control.fly_setpoints_mission:main',
            'fly_setpoints_takephoto_mission = px4_control.fly_setpoints_takephoto_mission:main',
            'obstacle_avoidance_static_gps = px4_control.obstacle_avoidance_static_gps:main',
            'obstacle_avoidance_static_offboard = px4_control.obstacle_avoidance_static_offboard:main',
            'obstacle_hold_gps = px4_control.obstacle_hold_gps:main',
        ],
    },
)