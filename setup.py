from setuptools import setup, find_packages

setup(
    name="grok_utils",
    version="0.1.0",
    packages=find_packages(),
    scripts=["scripts/train.py"],
    install_requires=[
        "torch",
        "transformers",
        "matplotlib",
        "tqdm",
        "scikit_learn",
        "blobfile",
    ],
    python_requires=">=3.6",
)
