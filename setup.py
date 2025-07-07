from setuptools import setup, find_packages

setup(
    name="kaggle_common",
    version="0.1.0",
    packages=find_packages(include=["common", "common.*"]),    # this will pick up common/ and utils/
)
