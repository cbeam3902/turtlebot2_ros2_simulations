from setuptools import find_packages, setup

package_name = 'turtlebot2_riosar'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/' + package_name, [
            package_name+'/riosar_concept.py',
            package_name+'/riosar.py',
            package_name+'/lidar_tdbp.py',
            package_name+'/__init__.py'
        ])
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Chris Beam',
    maintainer_email='cbeam18@charlotte.edu',
    description='RIO-SAR for the turtlebot2 https://ieeexplore.ieee.org/document/10739367',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'riosar_concept = turtlebot2_riosar.riosar_concept:main',
            'riosar = turtlebot2_riosar.riosar:main',
            'lidar_tdbp = turtlebot2_riosar.lidar_tdbp:main'
        ],
    },
)
