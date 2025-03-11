"""
API Documentation

Base URL: http://localhost:8000

1. Test API
   Endpoint: /test
   Method: GET, POST
   Description: Test endpoint to verify API functionality and server status
   Response: JSON containing:
   {
     "status": "operational",
     "version": "1.0.0",
     "timestamp": "ISO-8601 timestamp",
     "endpoints": {
       "test": "/test",
       "predict": "/predict",
       "web_predict": "/web/predict",
       "status": "/web/status/<task_id>",
       "download": "/download/<token>"
     },
     "models": {
       "yolo_n": boolean,
       "yolo_s": boolean,
       "yolo_m": boolean,
       "mobilenet": boolean,
       "vgg16": boolean
     },
     "directories": {
       "run_output": boolean,
       "run_extract": boolean,
       "input": boolean,
       "models": boolean
     },
     "cuda": {
       "available": boolean,
       "device_count": int,
       "device_name": string
     },
     "system": {
       "cpu_percent": float,
       "memory_percent": float,
       "disk_percent": float
     }
   }
   If POST method:
   - Includes "echo" field with POSTed JSON data
   - Includes "files" array with details of uploaded files if present

2. Direct Processing API
   Endpoint: /predict
   Method: POST
   Description: Synchronously processes uploaded files and returns results immediately
   Request:
   - Content-Type: multipart/form-data or application/json
   Parameters:
   - files: One or more files to process
     - For input_type=0: Single ZIP file with JPG/PNG/JPEG + JGW files, or individual JPG/PNG/JPEG + JGW files
     - For input_type=1: Either ZIP file with GeoTIFF files, or individual GeoTIFF (.tif, .tiff) files
   - input_type: String, '0' for image data (JPG/PNG + JGW), '1' for GeoTIFF data
   - classification_threshold: String, default '0.35'
   - prediction_threshold: String, default '0.5'
   - save_labeled_image: String, default 'false'
   - output_type: String, default '0' (0=JSON, 1=TXT)
   - yolo_model_type: String, default 'n' (n/s/m)
   Response: 
   - If Accept: application/json:
     {
       "status": "success",
       "message": "Processing completed",
       "output_path": string
     }
   - Otherwise: ZIP file containing results
   Error Response:
   {
     "error": string
   }

3. Web Processing API (Queued)
   Endpoint: /web/predict
   Method: POST
   Description: Asynchronously processes files with progress tracking
   Request:
   - Content-Type: multipart/form-data or application/json
   Parameters:
   - files: One or more files to process
     - For input_type=0: Single ZIP file with JPG/PNG/JPEG + JGW files, or individual JPG/PNG/JPEG + JGW files
     - For input_type=1: Either ZIP file with GeoTIFF files, or individual GeoTIFF (.tif, .tiff) files
   - input_type: String, '0' for image data (JPG/PNG + JGW), '1' for GeoTIFF data
   - classification_threshold: String, default '0.35'
   - prediction_threshold: String, default '0.5'
   - save_labeled_image: String, default 'false'
   - output_type: String, default '0' (0=JSON, 1=TXT)
   - yolo_model_type: String, default 'n' (n/s/m)
   Response: 
   {
     "task_id": string,
     "message": "Task queued successfully"
   }
   Error Response (503):
   {
     "error": "Server is busy. Please try again later."
   }

4. Task Status
   Endpoint: /web/status/<task_id>
   Method: GET
   Description: Get task status and progress
   Response:
   {
     "percentage": 0-100,           # Progress percentage
     "log": string,                 # Current stage or status message
     "has_detections": boolean,     # Whether any detections were found (for completed tasks)
     "download_token": string,      # Included when task is completed
     "error": string                # Included when task has failed or been cancelled
   }
   
   Notes:
   - For queued tasks, the log will contain "Waiting in queue (position X)"
   - For processing tasks, the log will show the current processing stage
   - For completed tasks, a download_token will be included
   - The has_detections flag indicates whether any object detections were found
   
   Error Response (404):
   {
     "error": "Task not found"
   }

5. Download Results
   Endpoint: /download/<token>
   Method: GET
   Description: Download processed results using token
   Response: 
   - Content-Type: application/zip
   - Content-Disposition: attachment; filename=result.zip
   - Content-Length: file size in bytes
   
   Token Verification Process:
   - Tokens are generated using JWT (JSON Web Tokens)
   - Token payload includes session_id and task_id
   - Tokens expire after 2 hours (MAX_TOKEN_AGE_HOURS)
   - Tokens are verified using auth_handler.verify_download_token
   - ZIP file integrity is verified before sending
   
   Error Responses:
   - 401: Invalid token or token payload
   - 404: Task or ZIP file not found
   - 500: ZIP file corrupted or empty

6. Cancel Task
   Endpoint: /web/cancel/<task_id>
   Method: POST
   Description: Cancel a running or queued task
   Response:
   {
     "status": "success",
     "message": "Task cancelled successfully"
   }
   
   Cancellation Process:
   - Sets cancellation flag on task
   - Removes task from processing queue if queued
   - Terminates execution if task is processing
   - Cleans up input/output folders for the task
   
   Error Response (404):
   {
     "error": "Task not found or cannot be cancelled"
   }

7. Server Status
   Endpoint: /server-status
   Method: GET
   Description: Get current server status and statistics
   Response: JSON containing:
   {
     "uptime_seconds": float,             # Server uptime in seconds
     "start_time": ISO-8601 timestamp,    # When server started
     "max_concurrent_tasks": int,         # Maximum allowed concurrent tasks
     "max_queue_size": int,               # Maximum allowed queue size
     "memory_usage_mb": float,            # Current memory usage in MB
     "cpu_usage_percent": float,          # Current CPU usage (0-100%)
     "active_tasks": int,                 # Total number of active tasks
     "processing_tasks": int,             # Number of tasks currently processing
     "queue_size": int,                   # Number of tasks in queue
     "queued_task_ids": [string],         # IDs of queued tasks
     "processing_task_ids": [string],     # IDs of processing tasks
     "total_tasks": int,                  # Total tasks processed
     "completed_tasks": int,              # Successfully completed tasks
     "failed_tasks": int,                 # Failed tasks
     "cancelled_tasks": int               # Cancelled tasks
   }

Additional Information:
- Maximum file size: 5GB
- CORS enabled for all origins with credentials support
- Preflight requests cached for 24 hours
- Supports methods: GET, POST, PUT, DELETE, OPTIONS, PATCH
- Background tasks:
  - Cleanup thread for removing old files
  - Queue processor thread for handling tasks
- Error responses include stack traces in development mode
- All endpoints support error handling with appropriate status codes
- File operations are logged for debugging and monitoring
- ZIP file integrity is verified before download

File Upload Rules and Processing:
1. For Image Data (input_type=0):
   - Supported file types: ZIP, JPG, JPEG, PNG, JGW
   - Must provide either:
     a) A single ZIP file containing JPG/PNG/JPEG + JGW files, or
     b) Individual JPG/PNG/JPEG + JGW files
   - System extracts contents from ZIP files if provided
   - System copies individual files to the processing directory
   - Hidden files starting with "._" are ignored

2. For GeoTIFF Data (input_type=1):
   - Supported file types: ZIP, TIF, TIFF
   - Must provide either:
     a) A single ZIP file containing GeoTIFF files, or
     b) Individual GeoTIFF files
   - System extracts contents from ZIP files if provided
   - System copies individual files to the processing directory
   - Hidden files starting with "._" are ignored

File Management:
- Each task creates a unique session ID: timestamp_uuid
- Files are uploaded to: input/{session_id}/
- Processing results saved to: run/output/{session_id}/
- Results are compressed into a ZIP file after processing
- Original directories are deleted after successful compression
- ZIP file integrity is verified after creation

Parameter Validation:
- input_type: Must be '0' (Image Data) or '1' (GeoTIFF Data)
- classification_threshold: String representation of float, default '0.35'
- prediction_threshold: String representation of float, default '0.5'
- save_labeled_image: String 'true' or 'false', default 'false'
- output_type: String '0' (JSON) or '1' (TXT), default '0'
- yolo_model_type: String 'n', 's', or 'm', default 'n'

Task Queue Management:
- Tasks are queued with a position number when server is busy
- Task position is updated as other tasks complete or are cancelled
- Maximum concurrent tasks is enforced by task_handler.MAX_CONCURRENT_TASKS
- Maximum queue size is enforced by task_handler.MAX_QUEUE_SIZE
"""