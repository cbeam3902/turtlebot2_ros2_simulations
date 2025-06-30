from setuptools import find_packages, setup

package_name = 'turtlebot2_ros2_teleop'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/' + package_name, [package_name+'/teleop_keyboard.py', package_name+'/__init__.py'])
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Chris Beam',
    maintainer_email='cbeam18@charlotte.edu',
    description='Teleop keyboard for Turtlebot2',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'teleop_keyboard = turtlebot2_ros2_teleop.teleop_keyboard:main'
        ],
    },
)
