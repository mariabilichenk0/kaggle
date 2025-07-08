from setuptools import setup, find_packages

setup(
    name="kaggle_common",
    version="0.1.0",
    packages=find_packages(include=["common", "common.*"]),    # this will pick up common/ and utils/
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn>=1.0",
        "category_encoders>=2.6.0",   
        # …other deps…
    ],
)
