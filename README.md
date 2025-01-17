# SightLink

SightLink is a computer vision system designed to detect and georeference buildings in aerial imagery. It processes both Digimap and custom aerial imagery, providing oriented bounding boxes with geographical coordinates. The system uses a combination of image segmentation, YOLO-based detection, and georeferencing to accurately identify and locate buildings in aerial photographs.

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

- Supports both Digimap and custom aerial imagery
- Automatic extraction and handling of zip archives
- Building detection using YOLO-based models (multiple variants available)
- Automatic georeferencing of detected buildings
- Multiple output formats (JSON/TXT)
- Progress tracking with detailed progress bars
- Organized output with timestamped directories
- Handles both single files and batch processing
- Automatic cleanup of temporary files
- Analysis tools for comparing detection results

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
git clone https://github.com/UCL-SightLink/SightLink-Main.git
cd SightLink-Main
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

```bash
pip install -r requirements.txt
```

4. Set up model files:

- Download and place YOLO model files in the `models/` directory:
  - `yolo-n.pt` (nano model)
  - Other model variants as needed

5. Create required directories:

```bash
mkdir -p input run/output
```

## Quick Start

1. Place input files in the `input` directory:

   - For Digimap data: Place zip files directly downloaded from DigiMap
   - For custom data: Place .jpg/.jgw files or zip archives
2. Run the system:

```bash
python run.py
```

## Usage Guide

### Basic Usage

```python
from main import execute

execute(
    uploadDir="input",           # Input directory
    inputType="0",              # "0" for Digimap, "1" for custom
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
      [[lon1,lat1], [lon2,lat2], [lon3,lat3], [lon4,lat4]],  # Building 1
      [[lon1,lat1], [lon2,lat2], [lon3,lat3], [lon4,lat4]]   # Building 2
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
SightLink-Main/
├── classificationScreening/    # Building classification module
│   ├── Classify.py            # Main classification logic
│   └── utils/                 # Classification utilities
├── imageSegmentation/         # Image segmentation modules
│   ├── boundBoxSegmentation.py       # Bounding box segmentation
│   └── classificationSegmentation.py  # Classification segmentation
├── models/                    # YOLO model files
│   ├── yolo-n.pt             # Nano model
│   └── MobileNetV3_state_dict_big_train.pth     # Classification Model
├── georeference/             # Georeferencing utilities
│   └── Georeference.py       # Coordinate conversion functions
├── utils/                    # Utility functions
│   ├── extract.py           # File extraction handling
│   └── analyze.py           # Result analysis tools
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

   - Handles Digimap zip files and custom inputs
   - Filters system files and unsupported formats
   - Organizes files for processing
2. **Image Segmentation**

   - Segments large aerial images
   - Prepares chunks for classification
   - Optimizes for detection accuracy
3. **Building Detection**

   - Uses selected YOLO model variant
   - Applies confidence thresholds
   - Supports multiple model types for different performance/accuracy trade-offs
4. **Georeferencing**

   - Converts pixel coordinates to geographical coordinates
   - Uses .jgw world files for accurate mapping
   - Handles coordinate system transformations
5. **Output Generation**

   - Creates timestamped directories
   - Generates selected output format
   - Optionally saves labeled images
   - Cleans up temporary files

### Performance Optimization

- GPU acceleration for faster processing
- Memory-efficient batch processing
- Progress tracking with estimated times
- Configurable model selection for speed/accuracy balance

## Troubleshooting

Common issues and solutions:

- **GPU not detected**: Ensure CUDA toolkit is installed
- **Memory errors**: Reduce batch size or use nano model
- **Missing files**: Check input directory structure
- **Coordinate errors**: Verify .jgw file format

## Contributing

[Add Contributing Guidelines]

## License

[Add License Information]

## Acknowledgments

- YOLO for object detection framework
- Ultralytics for YOLOv8 implementation
- [Add other acknowledgments]
