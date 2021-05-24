# support3d
It is a module to help with 3D point cloud processing.
It is based on Open3D.

## Requirements
* Open3D 0.12.0.0
* scikit-learn
* shapely
* numpy 

## Usage
* `core/base.py` The class that summarizes commonly used point cloud processing.
* `viewer/rotated_view.py` Rotate the camera to capture the point cloud and save it as a GIF.
* `registration/ransac.py` RANSAC Algorithm to matching features.
* `registration/docking.py` Unique tool to align point clouds(registration) that do not match perfectly.

## ToDo
- [ ] Updates other tools
- [ ] Add the example file of how to you

