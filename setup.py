from setuptools import setup, find_packages

setup(
    name="max-cv",
    version="0.1.0",
    author="Brad Larson",
    author_email="larson@sunsetlakesoftware.com",
    description="MAX-CV: Accelerated image processing via MAX and Mojo",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/BradLarson/max-cv",
    packages=find_packages(),
    install_requires=[
        "pillow>=11.0.0,<12",
        "max>=25.5,<25.6",
    ],
    extras_require={
        "test": ["pytest>=8.3.2,<9"],
        "notebook": [
            "matplotlib>=3.9.0,<4",
            "notebook>=7.3.2,<8",
        ],
        "showcase": [
            "click>=8.1.7",
            "matplotlib>=3.9.0,<4",
        ],
        "video": [
            "click>=8.1.7",
            "opencv-python>=4.11.0,<5",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    package_data={
        "max_cv": ["operations_mojo/*.mojo"],
    },
    include_package_data=True,
)
