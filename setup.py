from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="hawc-analysis",
    version="1.0.0",
    author="HAWC Collaboration",
    author_email="rbabu@mtu.com",
    description="HAWC Gamma-ray Source Analysis Pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/HAWC/hawc-analysis",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.20",
        "pandas>=1.3",
        "scipy>=1.7",
        "pyyaml>=5.4",
        "click>=8.0",
        "matplotlib>=3.4",
        "astropy>=5.0",
        "healpy>=1.14",
        "threeml>=3.0",
        "astromodels>=4.0",
        "scikit-image>=0.18",
        "hawc_hal",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.12",
            "black>=21.0",
            "flake8>=3.9",
            "sphinx>=4.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "hawc-analysis=hawc_analysis.cli.main:cli",
        ]
    },
)
