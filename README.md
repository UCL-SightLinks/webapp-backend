# SightLinks Backend

This repository contains the backend for the SightLinks web application, an AI-powered system for detecting solar panels from aerial imagery.

## System Overview

The SightLinks backend is a Flask-based API that processes aerial images using computer vision and deep learning techniques to detect and analyze solar panels. The key components include:

- **Image Segmentation**: Processes large aerial images by dividing them into manageable chunks
- **Object Detection**: Uses YOLOv11 models (n/s/m variants) to detect solar panels with oriented bounding boxes
- **Classification**: Analyzes detected areas to verify solar panel presence
- **Georeferencing**: Processes georeferenced images (.jgw/.tfw files or GeoTIFF) to provide accurate location data
- **Task Queue**: Handles asynchronous processing of submitted tasks
- **API Endpoints**: Provides REST endpoints for submitting, monitoring, and retrieving results

## Deployment Instructions

### Prerequisites

- Python 3.8+
- GDAL 3.6.4
- PyTorch 2.0.0+
- CUDA-capable GPU (recommended for production)

### Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/UCL-SightLinks/webapp-backend.git
   cd webapp-backend
   ```
2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
4. Create necessary directories:

   ```bash
   mkdir -p input run/output run/extract
   ```
5. Download required model files and place them in the `models` directory:

   - YOLOv11 models: yolo-n.pt, yolo-s.pt, yolo-m.pt
   - Classification models: MobileNetV3_state_dict_big_train.pth, VGG16_Full_State_Dict.pth

### Running in Development

For development purposes, you can run the application with:

```bash
python app.py
```

The server will start on http://localhost:5000 (default Flask port).

### Production Deployment

For production deployment, we recommend using Gunicorn:

1. Configure your environment:

   ```bash
   cp startup.txt.example startup.txt
   # Edit startup.txt as needed
   ```
2. Start the server using the startup script:

   ```bash
   chmod +x startup.sh
   ./startup.sh
   ```
3. Alternatively, run Gunicorn directly:

   ```bash
   gunicorn --bind 0.0.0.0:8000 \
            --timeout 600 \
            --workers 2 \
            --threads 4 \
            --log-level info \
            app:app
   ```

### Docker Deployment

A Dockerfile is included for containerized deployment:

1. Build the Docker image:

   ```bash
   docker build -t sightlinks-backend .
   ```
2. Run the container:

   ```bash
   docker run -p 8000:8000 -v /path/to/data:/app/input sightlinks-backend
   ```

## API Endpoints

- `GET /test` - Test endpoint to verify API functionality
- `POST /predict` - Synchronous prediction endpoint (waits for completion)
- `POST /web/predict` - Asynchronous prediction endpoint (returns task ID)
- `GET /web/status/<task_id>` - Get status of a submitted task
- `GET /download/<token>` - Download results using a token
- `POST /web/cancel/<task_id>` - Cancel a running task
- `GET /server-status` - Get server status information

## Configuration

Core application settings can be modified in `run.py`, including:

- `uploadDir` - Directory for uploaded files
- `inputType` - Type of input (0 for JPG/JGW, 1 for TIF)
- `classificationThreshold` - Threshold for classification model
- `predictionThreshold` - Threshold for YOLO detection model
- `saveLabeledImage` - Whether to save labeled images
- `outputType` - Output format (0 for JSON, 1 for TXT)
- `yoloModelType` - YOLO model variant ('n', 's', or 'm')

## Troubleshooting

- Check the application logs for error messages
- Verify CUDA availability with the `/test` endpoint
- Ensure all required model files exist in the `models` directory
- Check disk space, as large image files can consume significant storage
