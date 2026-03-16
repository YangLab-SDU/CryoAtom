import os
import sys

from setuptools import setup, find_packages

import CryoAtom2


setup(
    name="CryoAtom2",
    author='Baoquan Su',
    author_email='202217127@mail.sdu.edu.cn',
    entry_points={
        "console_scripts": [
            "cryoatom = CryoAtom2.__main__:main",
        ],
    },
    packages=find_packages(),
    package_data={
        '': ['./*.json','./utils/*','./utils/*.py','./utils/*.txt' ,'./RUNet/*.py', './CryoNet/*.py', './checkpoint/*.pth'],
    },
    include_package_data=True,
    version=CryoAtom2.__version__,
)
