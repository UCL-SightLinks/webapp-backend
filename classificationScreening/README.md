# Zebra Crossing Classification for Screening Input Data

This module implements a binary classification model for screening satellite image tiles based on whether they are likely to contain a zebra crossing. The inference functions are included (The pretrained model is available but exceeds the size that can be uploaded) but the functions for the processing and loading of the dataset are also included, to allow for fine tuning and further training of the models.

The legacy versions of the classification are included for posterity and progress reporting, but are not included in the dependencies as they are not used.

## Overview
### Current Supported Model Architectures
```
VGG16 - modified to output binary predictions (requires resizing input images to 224x224) 

ResNet18 - modified to output binary predictions (requires resizing input images to 256x256)
```
### Data and the dataset
The classifier takes input images in numpy array format, and returns a probability that represents the confidence it has in the image containing a crosswalk. This probability can be thresholded to screen input data before passing it to the object detection system. It is recommended that a low probability threshold (E.g. ~0.35) is used.

The project uses a custom dataset class, CrosswalkDataset, which is part of the ClassUtils module. This module is responsible for data loading and preprocessing for training purposes. You will need to have the dataset structured properly under __zebra_annotations/classification_data__, of which some examples are available. Each label consists of a text file containing 0 for a negative case, and 1 for a positive case.

Another option is to use the dataset structure expected by the Object Detection model, and then convert that data to the format expected by ClassUtils. To do this, put the images in the zebra_images folder and their associated label in txt_annotations, ensuring that they are linked by the same stem name (name excluding file extensions)


### Example Output from the **InferAndDisplay** function

![Example Classification](DisplayImages/Screenshot%202025-01-10%20at%2014.41.27.png)

![Example Classification](DisplayImages/Screenshot%202025-01-10%20at%2014.41.41.png)

![Example Classification](DisplayImages/Screenshot%202025-01-10%20at%2014.42.09.png)


### Dependencies
Required Packages
```
torch>=2.0.0
numpy>=1.24.0
PyYAML>=6.0.1
matplotlib>=3.7.0
Pillow>=10.0.0
shapely>=2.0.0
shutils>=0.1.0
```

They can be installed using pip using the following commands
```
pip install torch>=2.0.0
pip install ultralytics>=8.0.0
pip install opencv-python>=4.8.0
pip install numpy>=1.24.0
pip install PyYAML>=6.0.1
pip install matplotlib>=3.7.0
pip install Pillow>=10.0.0
pip install shapely>=2.0.0
pip install shutils>=0.1.0
```
