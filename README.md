# SightLinks

SightLinks is a computer vision system designed to detect and georeference crosswalks in aerial imagery. It processes .jpg (or .jpeg, .png) with their corresponding .jgw file and .tif files, providing oriented bounding boxes with latitude and longitude coordinates. The system uses a combination of image segmentation, mobileNet detection, YOLO-based detection, georeferencing, and filtering to accurately identify and locate crosswalks in aerial photographs.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## Features

- Supports .jpg with .jgw files and .tif files
- Automatic extraction and handling of zip input files
- Crosswalk detection using YOLO-based models (multiple variants available)
- Automatic georeferencing and filtering of detected crosswalks
- Multiple output formats (JSON/TXT)
- Progress tracking with detailed progress bars
- Organized output with timestamped directories
- Handles both single files and batch processing
- Visualization tool for comparing detection results.

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Git
- CUDA-capable GPU recommended for faster processing
- Sufficient disk space for image processing
- Required Python packages (installed via requirements.txt)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/UCL-SightLinks/SightLinks-Main.git
cd SightLinks-Main
```

2. Create and activate a virtual environment:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

3. Install dependencies:

For Windows and Linux Machines:
```bash
sudo apt update
sudo apt install gdal-bin libgdal-dev
pip install -r requirements.txt
```
For MacOS machines:
```bash
brew update
brew install gdal
pip install -r requirements.txt
```

## Quick Start

1. Place input files in the `input` directory:

   - For .jpg/.jgw data: Place zip files containing .jpg/.jgw files (e.g. from Digimap), or directly place the .jpg/.jgw files.
   - For .tif data: Place zip files containing .tif files, or directly place the .tif files.
2. Run the system:

```bash
python run.py
```

## Usage Guide

### Basic Usage

```python
// This snippet of code is from run.py
from main import execute

execute(
    uploadDir="input",           # Input directory
    inputType="0",              # "0" for .jpg/.jgw, "1" for .tif files
    classificationThreshold=0.35,
    predictionThreshold=0.5,
    saveLabeledImage=False,
    outputType="0",             # "0" for JSON, "1" for TXT
    yolo_model_type="n"         # "n" for nano model
)
```

### Output Formats

1. JSON Format (output_type="0"):

```json
[
  {
    "image": "image_name.jpg",
    "coordinates": [
      [[lon1,lat1], [lon2,lat2], [lon3,lat3], [lon4,lat4]],  # crosswalk 1
      [[lon1,lat1], [lon2,lat2], [lon3,lat3], [lon4,lat4]]   # crosswalk 2
    ]
  }
]
```

2. TXT Format (output_type="1"):

- One file per original image
- Each line represents one building:

```
lon1,lat1 lon2,lat2 lon3,lat3 lon4,lat4
```

### Output Directory Structure

```
run/output/YYYYMMDD_HHMMSS/  # Timestamp-based directory
├── output.json              # If JSON output selected
├── image_name.txt          # If TXT output selected (one per image)
└── labeledImages/          # Optional: Images with visualized detections
```

## Project Structure

```
SightLinks-Main/
├── classificationScreening/    # Building classification module
│   ├── classify.py            # Main classification logic
│   └── utils/                 # Classification utilities
├── imageSegmentation/         # Image segmentation modules
│   ├── boundBoxSegmentation.py       # Bounding box segmentation
│   └── classificationSegmentation.py  # Classification segmentation
├── models/                    # YOLO model files
│   ├── yolo-n.pt             # Nano model
│   └── mn3_vs55.pth     # Classification Model
├── georeference/             # Georeferencing utilities
│   └── Georeference.py       # Coordinate conversion functions
├── utils/                    # Utility functions
│   ├── extract.py           # File extraction handling
│   ├── compress.py           # File compression handling
│   ├── filterOutput.py           # Filters bounding boxes to remove duplicates
│   ├── saveToOutput.py           # Saves stored coordinates to output file
│   └── visualize.py           # Result analysis tools
├── run/                      # Runtime directories
│   └── output/              # Timestamped outputs
├── input/                   # Input file directory
├── requirements.txt         # Python dependencies
├── main.py                 # Main execution module
└── run.py                  # Quick start script
```

## Technical Details

### Processing Pipeline

1. **File Extraction**

   - Handles .jpg/.jgw files and .tif files
   - Filters system files and unsupported formats
   - Organizes files for processing
2. **Image Segmentation**

   - Segments large aerial images
   - Prepares chunks for classification
3. **Image classification**

   - Process the segmented images using the classification model
   - Returns True if the model's confidence is greater than a certain threshold
4. **Image Segmentation**

   - Re-segment the images based on the rows and columns of interest (where the classification model returns True)
   - Prepares chunks for classification
5. **Crosswalk Detection**

   - Uses selected YOLO model variant
   - Applies confidence thresholds
   - Supports multiple model types for different performance/accuracy trade-offs
6. **Georeferencing**

   - Converts pixel coordinates to geographical coordinates
   - Uses .jgw world files or data stored in .tif files for accurate mapping 
   - Handles coordinate system transformations
7. **Filtering**

   - Removes duplicate bounding boxes by using Non-Maximum Suppression
8. **Output Generation**

   - Creates timestamped directories
   - Generates selected output format
   - Optionally saves labeled images

### Performance Optimization

- GPU acceleration for faster processing
- Filter by row and column for a more optimised filter
- Progress tracking with a progress bar
- Configurable model selection for speed/accuracy balance

## Troubleshooting

Common issues and solutions:

- **Memory errors**: Reduce batch size or use nano model
- **Missing files**: Check input directory structure