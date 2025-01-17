# Oriented Bounding Box Detection for Zebra Crossings

This project implements an oriented bounding box (OBB) detector for zebra crossings using YOLOv8. The model is trained to detect zebra crossings in aerial/satellite imagery and outputs oriented bounding boxes that accurately capture the orientation of the crossings.

## Project Structure

```
.
├── dataset/              # Training and validation dataset
│   ├── images/          # Images for training and validation
│   └── labels/          # Labels in YOLO OBB format
├── test/                # Testing related files
│   ├── test-data/images/          # Test images
│   ├── model.pt        # Trained model weights
│   └── test.py         # Testing script
├── utils/               # Utility scripts
│   ├── data_sorting.py      # Data preparation utilities
│   ├── split_dataset.py     # Dataset splitting script
│   └── visualize_labels_xy.py # Label visualization
└── train/               # Training related files
    ├── dataset.yaml     # Dataset configuration
    └── train.py         # Training script
```

## Testing the Model

### Quick Start

1. Place your test images in the `test-data/images` directory
2. Run the test script:

```bash
cd test
python test.py
```

3. Results will be saved in `test/runs/obb/predict/`

### Test Outputs

The test script generates several outputs:

- Predicted images with visualizations in `test/runs/obb/predict/`
- Label files in YOLO OBB format in `test/runs/obb/predict/labels/`
- Each detection includes:
  - Class ID (0 for zebra crossing)
  - Confidence score
  - Oriented bounding box coordinates

### Label Format

The model outputs labels in YOLO OBB format. Each line in the label file represents one detection:

```
<class_id> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>
```

Where:

- `class_id`: Integer class identifier (0 for zebra crossing)
- `x1,y1` to `x4,y4`: Normalized coordinates (0-1) of the four corners of the oriented bounding box

Example label:

```
0 0.2514 0.3851 0.3126 0.3842 0.3117 0.4023 0.2505 0.4032
```

This represents:

- Class ID: 0 (zebra crossing)
- Point 1: (0.2514, 0.3851) - Top-left corner
- Point 2: (0.3126, 0.3842) - Top-right corner
- Point 3: (0.3117, 0.4023) - Bottom-right corner
- Point 4: (0.2505, 0.4032) - Bottom-left corner

Points are ordered clockwise starting from the top-left corner. All coordinates are normalized to the range [0,1]:

- x coordinates are normalized by image width
- y coordinates are normalized by image height

To convert to pixel coordinates:

```python
x_pixel = x_normalized * image_width
y_pixel = y_normalized * image_height
```

## Training

### Dataset Preparation

1. Organize your dataset in the following structure:

```
dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

2. Configure `train/dataset.yaml` with your dataset paths
3. Run training:

```bash
python train/train.py
```

## Requirements

### Environment Setup
```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

### Dependencies
Required packages:
```txt
torch>=2.0.0
ultralytics>=8.0.0
opencv-python>=4.8.0
numpy>=1.24.0
PyYAML>=6.0.1
matplotlib>=3.7.0
Pillow>=10.0.0
```

Quick install:
```bash
# Install all dependencies
pip install -r requirements.txt

# Or install individually
pip install torch>=2.0.0
pip install ultralytics>=8.0.0
pip install opencv-python>=4.8.0
pip install numpy>=1.24.0
pip install PyYAML>=6.0.1
pip install matplotlib>=3.7.0
pip install Pillow>=10.0.0
```

### CUDA Support (Optional)
For GPU acceleration:
```bash
# Install PyTorch with CUDA support
pip install torch>=2.0.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```

## Model Performance

- The model is trained to detect zebra crossings in aerial/satellite imagery
- Outputs oriented bounding boxes that capture crossing orientation
- Confidence threshold can be adjusted in test scripts (default: 0.25)
