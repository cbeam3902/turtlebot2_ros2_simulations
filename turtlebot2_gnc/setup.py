from setuptools import find_packages, setup

package_name = 'turtlebot2_gnc'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/' + package_name, [
            package_name+'/gnc.py', 
            package_name+'/random_explorer.py', 
            package_name+'/__init__.py',
            package_name+'/urad_classifier.pt'
        ])
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Chris Beam',
    maintainer_email='cbeam18@charlotte.edu',
    description='Simple GNC for Turtlebot2',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'gnc = turtlebot2_gnc.gnc:main',
            'random_explorer = turtlebot2_gnc.random_explorer:main'
        ],
    },
)
