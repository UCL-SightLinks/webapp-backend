# API Documentation

## Base URL
`http://localhost:5010`

## Endpoints

### 1. Direct Processing (`POST /api/predict`)
Synchronously processes images and returns results immediately.

#### Request
```json
Content-Type: multipart/form-data
{
    "file": "ZIP file",
    "input_type": "0",                    // optional
    "classification_threshold": "0.35",    // optional
    "prediction_threshold": "0.5",        // optional
    "save_labeled_image": "false",        // optional
    "output_type": "0",                   // optional
    "yolo_model_type": "n"               // optional
}
```

#### Response
- Success (200): ZIP file containing results
- Error (400/500): `{"error": "message"}`

### 2. Web Processing (`POST /web/predict`)
Asynchronously processes images with progress tracking.

#### Request
Same as Direct Processing

#### Response
- Success (200): `{"task_id": "uuid", "message": "Task queued successfully"}`
- Error (400/500/503): `{"error": "message"}`

### 3. Task Status (`GET /web/status/<task_id>`)
Get processing status and progress.

#### Response
```json
{
    "status": "queued|processing|completed|failed",
    "progress": 0-100,
    "stage": "current_stage",
    "download_token": "jwt-token"  // only when completed
}
```

Processing Stages:
- Initializing (5%)
- Extracting files (10%)
- Initializing model (20%)
- Processing images (40%)
- Segmenting images (60%)
- Creating bounding boxes (70%)
- Saving results (80%)
- Creating ZIP file (90%)
- Completed (100%)

### 4. Download Results (`GET /download/<token>`)
Download processed results using token.

#### Response
- Success (200): ZIP file
- Error (401/404/500): `{"error": "message"}`

## Quick Start

```python
import requests
import time

# 1. Submit task
files = {'file': open('images.zip', 'rb')}
data = {'input_type': '0', 'classification_threshold': '0.35'}
response = requests.post('http://localhost:5010/web/predict', files=files, data=data)
task_id = response.json()['task_id']

# 2. Track progress
while True:
    status = requests.get(f'http://localhost:5010/web/status/{task_id}').json()
    if status['status'] == 'completed':
        # 3. Download results
        token = status['download_token']
        result = requests.get(f'http://localhost:5010/download/{token}')
        with open('result.zip', 'wb') as f:
            f.write(result.content)
        break
    elif status['status'] == 'failed':
        print(f"Error: {status.get('error')}")
        break
    print(f"Progress: {status['progress']}% - {status['stage']}")
    time.sleep(1)
```

## Notes
- Only ZIP files accepted
- Max queue size: 10 tasks
- Results deleted after 2 hours (except ZIP files)
- All intermediate files are cleaned up after processing 