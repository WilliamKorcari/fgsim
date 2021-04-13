#!/usr/bin/env python

"""The setup script."""

# needed for build the cpp extension
# from glob import glob
# import numpy as np
# from pybind11.setup_helpers import Pybind11Extension
from setuptools import find_packages, setup

with open("docs/readme.rst") as readme_file:
    readme = readme_file.read()

with open("docs/history.rst") as history_file:
    history = history_file.read()


setup_requirements = ["pybind11==2.6.1", "numpy==1.19.4"]
install_requirements = [
    "awkward==1.2.2",
    "awkward0==0.15.5",
    "bump2version==1.0.1",
    "matplotlib==3.4.1",
    "msgpack==1.0.2",
    "neovim==0.3.1",
    "numba==0.53.1",
    "numpy==1.20.2",
    "omegaconf==2.0.6",
    "Pillow==8.2.0",
    "pretty-errors==1.2.19",
    "prettytable==2.1.0",
    "pybind11==2.6.2",
    "PyYAML==5.4.1",
    "setuptools==56.0.0",
    "six==1.15.0",
    "torch==1.8.1",
    "torch-cluster==1.5.9",
    "torch-geometric==1.7.0",
    "torch-scatter==2.0.6",
    "torch-sparse==0.6.9",
    "torch-spline-conv==1.2.1",
    "torchvision==0.9.1",
    "tox==3.23.0",
    "tqdm==4.60.0",
    "uproot==4.0.7",
    "uproot3==3.14.4",
]

extras = {
    "test": [
        "coverage==5.5",
        "flake8==3.9.0",
        "ipython==7.22.0",
        "isort==5.8.0",
        "black==20.8b1",
        "jedi==0.18.0",
        "pycodestyle==2.7.0",
        "pylint==2.7.4",
        "pynvim==0.4.3",
        "pytest==6.2.3",
        "pytest-runner==5.3.0",
    ],
}

ext_modules = [
    # Pybind11Extension(
    #     "geomapper",
    #     sources=sorted(glob("fgsim/geo/*.cpp")),
    #     include_dirs=[
    #         np.get_include(),
    #         "fgsim/geo/libs/xtensor/include",
    #         "fgsim/geo/libs/xtensor-python/include",
    #         "fgsim/geo/libs/xtl/include",
    #     ],
    #     extra_compile_args=["-std=c99", "-Wno-error=vla"],
    # ),
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
