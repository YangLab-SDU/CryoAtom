import os
import sys

from setuptools import setup, find_packages

import CryoAtom


setup(
    name="CryoAtom",
    author='Baoquan Su',
    author_email='202217127@mail.sdu.edu.cn',
    entry_points={
        "console_scripts": [
            "cryoatom = CryoAtom.__main__:main",
        ],
    },
    packages=find_packages(),
    package_data={
        '': ['./*.json','./utils/*','./utils/*.py','./utils/*.txt' ,'./Unet/*.py', './CryNet/*.py', './checkpoint/*.pth'],
    },
    include_package_data=True,
    version=CryoAtom.__version__,
)
