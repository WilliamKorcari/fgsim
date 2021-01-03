#!python

"""The setup script."""

from setuptools import setup, find_packages, Extension
from pybind11.setup_helpers import Pybind11Extension
from glob import glob

with open("docs/readme.rst") as readme_file:
    readme = readme_file.read()

with open("docs/history.rst") as history_file:
    history = history_file.read()


setup_requirements = ["pybind11==2.6.1", "numpy==1.19.4"]
install_requirements = [
    "pybind11==2.6.1",
    "numpy==1.19.4",
    "wheel==0.33.6",
    "pretty_errors==1.2.19",
    "torch==1.7.1",
    "torchvision==0.8.2",
    "imageio==2.9.0",
    "bump2version==0.5.11",
]

extras = {
    "test": [
        "tox==3.14.0",
        "tox-wheel==0.6.0",
        "pytest==4.6.5",
        "pytest-runner==5.1",
        "coverage==5.3.1",
    ],
}


import numpy as np
import os

ext_modules = [
    Pybind11Extension(
        "geomapper",
        sources=sorted(glob("fgsim/geo/*.cpp")),
        include_dirs=[
            np.get_include(),
            "fgsim/geo/libs/xtensor/include",
            "fgsim/geo/libs/xtensor-python/include",
            "fgsim/geo/libs/xtl/include",
        ]
        # extra_compile_args=["-std=c99", "-Wno-error=vla"],
    ),
]


setup(
    author="Anonymous",
    author_email="mova@users.noreply.github.com",
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python :: 3.8",
    ],
    description="Fast simulation of the HGCal using neural networks.",
    entry_points={
        "console_scripts": [
            "fgsim=fgsim.__main__.py:main",
        ],
    },
    install_requires=install_requirements,
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="fgsim",
    name="fgsim",
    packages=find_packages(
        include=["fgsim"], exclude=["fgsim/geo/libs/.*", "xtensor-python"]
    ),
    setup_requires=setup_requirements,
    ext_modules=ext_modules,
    extras_require=extras,
    url="https://github.com/mova/fgsim",
    version="0.1.0",
    zip_safe=False,
)
