## Overview
This project aims to classify 3D shapes from the ModelNet10 dataset using neural networks. ModelNet10 is a widely used benchmark dataset for 3D object recognition, containing 10 categories of objects represented in 3D CAD models.

## Dataset
The ModelNet10 dataset can be downloaded [here](http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip). The dataset consists of 10 categories:
- Bathtub
- Bed
- Chair
- Desk
- Dresser
- Monitor
- Nightstand
- Sofa
- Table
- Toilet

## Dataset Conversion
Run `python preprocessing.py` to transform the CAD models in .off format to voxelgrids, saved as .pt files, using the same folder structure of `ModelNet10`. The converted dataset is stored as `ModelNet10Voxel`.

The script can be run as is, or configured to accomodate for your computer capabilities. 

## Approach
A 3D Convolutional Neural Network is implemented for now.



Using Simple3DCNN, with just 600k parameters we obtained an accuracy of 92.4%. Not bad. Loss 0.214. 