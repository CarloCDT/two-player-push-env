"""Setup configuration for two-player-push-env package."""

from pathlib import Path

from setuptools import find_packages, setup

# Read README with proper encoding
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="two-player-push-env",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "gymnasium>=0.29.0,<0.30.0",
        "numpy>=1.21.0,<2.0.0",
        "pytest>=7.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "isort>=5.0.0",
            "flake8>=4.0.0",
            "flake8-docstrings>=1.6.0",
            "pre-commit>=3.3.3",
        ],
    },
    author="Carlo di Francescantonio",
    author_email="carlo@factored.ai",
    description="A two-player competitive pushing game environment for RL",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/CarloCDT/two-player-push-env",
    project_urls={
        "Bug Tracker": "https://github.com/CarloCDT/two-player-push-env/issues",
        "Documentation": "https://github.com/CarloCDT/two-player-push-env#readme",
        "Source Code": "https://github.com/CarloCDT/two-player-push-env",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
