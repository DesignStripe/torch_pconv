#!/usr/bin/env python3
import pathlib

from setuptools import setup

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

setup(
    name="torch_pconv",
    version="0.1.0",
    packages=["torch_pconv"],
    description="Faster and more memory efficient implementation of the Partial Convolution 2D"
                " layer in PyTorch equivalent to the standard NVidia implem.",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/DesignStripe/torch_pconv",
    author="Samuel Prevost",
    author_email="samuel.prevost@pm.me",
    licence="BSD",
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    include_package_data=True,
    install_requires=["torch", "tensor_type", "pshape"],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
)
