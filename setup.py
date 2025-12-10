from setuptools import setup, find_packages

setup(
    name="melodic-metadata-massacrer",
    version="2.0.0",
    description="The audio anonymizer that makes AI detectors cry",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Security Research Team",
    url="https://github.com/research/mmm",
    packages=find_packages(),
    install_requires=[
        "click>=8.1.0",
        "mutagen>=1.47.0",
        "librosa>=0.10.1",
        "pydub>=0.25.1",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "soundfile>=0.12.1",
        "colorama>=0.4.6",
        "rich>=13.0.0",
        "PyYAML>=6.0",
    ],
    entry_points={
        "console_scripts": [
            "mmm=mmm.cli:cli",
        ],
    },
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Security",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
    ],
)