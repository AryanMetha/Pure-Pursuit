from setuptools import find_packages, setup
import os
from glob import glob 
package_name = 'pp_trial'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        #launch files
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),

        # Include config (YAML) files
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),

        # Include URDF files
        (os.path.join('share', package_name, 'urdf'),
            glob('urdf/*.urdf')),

        # Include RViz config (optional)
        (os.path.join('share', package_name, 'config'),
            glob('config/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='aryan',
    maintainer_email='aryan100306@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pure_pursuit_node = pp_trial.pure_pursuit:main',
            'adaptive_pp_init = pp_trial.adaptive_pp:main',
            'adaptive_pure_pursuit_node= pp_trial.adaptive_implementation:main',
        ],
    },
)
