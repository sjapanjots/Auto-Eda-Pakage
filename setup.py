from setuptools import setup, find_packages

setup(
    name="flasheda",
    version="0.1.0",
    description="Constant-time EDA for any dataset size — one line, fixed time.",
    author="Your Name",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "rich>=12.0.0",
        "jinja2>=3.0.0",
        "matplotlib>=3.4.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)