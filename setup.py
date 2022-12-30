from distutils.core import setup

import setuptools

with open("README.md") as f:
    long_description = f.read()

extras_include = {
    "dev": [
        "docker-compose",
        "flake8",
        "black",
        "isort",
        "mypy",
        "pytest",
        "pytest-benchmark",
    ],
}

extras_include["all"] = [
    *extras_include["dev"],
]

setup(
    name="PyEdge",
    version="0.0.1",
    author="Ryan Smith",
    author_email="rsmith@plainsight.ai",
    url="https://github.com/rmsmith251/PyEdge",
    packages=setuptools.find_packages(),
    package_data={"pyedge": ["py.typed"]},
    include_package_data=True,
    description="Self-hosted inference server for object detection and tracking",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "click==8.0.4",
        "fsspec[gs]",
        "torch",
        "torchvision",
        "numpy",
        "opencv-python",
        "pydantic",
        "ftfy",
        "scipy",
        # "huggingface-hub",
        "timm",
        "fastapi",
        "uvicorn",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
