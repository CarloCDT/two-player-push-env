from setuptools import setup, find_packages

setup(
    name="two-player-push-env",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "gymnasium>=0.29.0",
        "numpy>=1.21.0",
    ],
    author="Carlo di Francescantonio",
    author_email="carlo@factored.ai",
    description="A two-player competitive pushing game environment for RL",
    long_description=open("README.md").read(),
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
