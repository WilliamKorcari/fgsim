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
    "pybind11==2.6.1",
    "wheel==0.33.6",
    "bump2version==0.5.11",
    "awkward==1.0.1",
    "flake8==3.8.4",
    "ipython==7.19.0",
    "isort==5.7.0",
    "jedi==0.17.2",
    "jupyter==1.0.0",
    "matplotlib==3.3.3",
    "neovim==0.3.1",
    "numba==0.52.0",
    "numpy==1.19.5",
    "omegaconf==2.0.6",
    "pretty-errors==1.2.19",
    "prettytable==2.0.0",
    "pybind11==2.6.1",
    "pylint==2.6.0",
    "pynvim==0.4.2",
    "PyYAML==5.4",
    "torch==1.7.1",
    "torchvision==0.8.2",
    "tox==3.21.1",
    "tqdm==4.56.0",
    "uproot==4.0.0",
    "uproot3==3.14.2",
    "psutil==5.8.0",
]

extras = {
    "test": [
        "pytest==4.6.5",
        "pytest-runner==5.1",
        "coverage==5.3.1",
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
