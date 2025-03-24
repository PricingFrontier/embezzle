from setuptools import setup, find_packages

setup(
    name="embezzle",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "statsmodels",
        "PyQt6",
        "matplotlib",
        "scikit-learn",
    ],
    author="GLM Modelling Team",
    author_email="your.email@example.com",
    description="A Python library for visualizing and building GLMs using statsmodels with PyQt",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/embezzle",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "embezzle=embezzle.ui.main_window:run_app",
        ],
    },
)
