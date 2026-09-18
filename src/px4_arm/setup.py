from setuptools import find_packages, setup

package_name = 'px4_arm'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kiwi',
    maintainer_email='lequyphuong1903@gmail.com',
    description='Minimal node that arms a PX4 vehicle over the uXRCE-DDS bridge.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'arm_node = px4_arm.arm_node:main',
        ],
    },
)
