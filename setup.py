from setuptools import find_packages, setup

setup(
    name="incontext_grokking",
    packages=find_packages(),
    version="0.1",
    install_requires=[
        "numpy",
        "torch",
        "scipy",
        "mod",
        "matplotlib",
        "blobfile",
    ],
)


