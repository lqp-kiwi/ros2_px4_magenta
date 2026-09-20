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
    license='ROWAN',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'offboard_arm = px4_control.offboard_arm:main',
            'takeoff_land = px4_control.takeoff_land:main',
        ],
    },
)
