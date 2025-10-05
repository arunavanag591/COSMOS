from setuptools import setup, find_packages

setup(
    name="cosmos-odor",
    version="1.0.1",
    description="A Data-Driven Probabilistic Time Series Simulator for Chemical Plumes",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Aruna Nag",
    author_email="arunavanag591@github.com",
    url="https://github.com/arunavanag591/COSMOS",
    packages=find_packages(),
    package_data={
        'cosmos': ['data/**/*'],
    },
    include_package_data=True,
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0", 
        "scipy>=1.7.0",
        "numba>=0.56.0",
        "h5py>=3.0.0",
    ],
    extras_require={
        'visualization': ['matplotlib>=3.5.0'],
        'dev': ['pytest>=6.0', 'black', 'flake8'],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: CC0 1.0 Universal (CC0 1.0) Public Domain Dedication",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    license="CC0-1.0",
    keywords="odor simulation chemical plumes environmental monitoring agent-based modeling",
)