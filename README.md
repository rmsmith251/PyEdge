# PyEdge
PyEdge is intended to be an easy to use, easy to set up ML server for Edge devices

# Why PyEdge
PyEdge is intended to allow for anyone to connect to a video feed and feed that stream through a pretrained model. The main use case for this would be home security and monitoring but is only limited to the use cases of Computer Vision. This is meant to be an open source alternative to current closed source implementations (such as Ring Doorbells by Amazon). This is intended to run on all consumer grade hardware as well as small edge devices like Nvidia Jetson. As this gets developed, I'll keep performance stats for a Jetson device as well as a midrange consumer PC and welcome others to test it on their PCs as well.

# Getting started
This is currently not on PyPI so to get started, clone the repo using
```
git clone git+ssh://git@github.com/rmsmith251/PyEdge
```
Then `cd` into the directory, set up your virtual environment using your preferred solution (I use PyEnv) and run
```
pip install .
```
If you want to develop on this library, install the repo with the `-e` flag to put it into editable mode, like so
```
pip install -e .
```

# Current limitations
This is still very much a work in progress so many items aren't implemented.

# TODO
- Implement the necessary processors for running the service
- Implement object detection models
- Implement object tracking for object detection
- Implement CLI
- Implement storage solution (cloud and local)
- Implement API (maybe)
- Figure out what else I need to do