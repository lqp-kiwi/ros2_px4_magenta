from setuptools import find_packages, setup

package_name = 'siyi_camera'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rowan',
    maintainer_email='rowan@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'take_photo = siyi_camera.take_photo:main',
            'qr_detect = siyi_camera.qr_detect:main',
            'qr_stream = siyi_camera.qr_stream:main',
            'record_video = siyi_camera.record_video:main',
        ],
    },
)