from setuptools import setup, find_packages

setup(
    name="pds",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    author="Shenyao Jin",
    author_email="shenyaojin@mines.edu",
    description="A package for solving the PDS equation",
    install_requires=[
        "numpy~=1.26.4",
        "matplotlib~=3.9.2",
        "scipy~=1.13.1",
        "setuptools~=75.1.0",
        "python-dateutil~=2.9.0post0",
        "pandas~=2.2.2",
        "h5py~=3.11.0",
        "plotly~=5.24.1"
    ],
)