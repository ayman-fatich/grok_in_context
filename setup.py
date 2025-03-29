from setuptools import find_package, setup

setup(
    name="incontext_grokking",
    packages=find_packages(),
    version="0.1",
    install_requires=[
        "numpy",
        "torsh",
        "scipy",
        "mod",
        "matplotlib",
    ],
)
