"""
API Documentation

Base URL: http://localhost:8000/api

1. Test API
   Endpoint: /test
   Method: GET, POST
   Description: Test endpoint to verify API functionality and server status
   Response: JSON containing:
   - Server status and version
   - Available endpoints
   - Model availability status
   - Directory structure status
   - CUDA availability
   - System resources (CPU, memory, disk usage)
   - Echo of POST data (if POST method)
   - File information (if files uploaded)

2. Direct Processing API
   Endpoint: /predict
   Method: POST
   Description: Synchronously processes uploaded files and returns results immediately
   Parameters:
   - file: ZIP file containing images (required)
   - input_type: Input type (0-1, default: 0)
   - classification_threshold: Classification threshold (0-1, default: 0.35)
   - prediction_threshold: Prediction threshold (0-1, default: 0.5)
   - save_labeled_image: Whether to save labeled images (true/false, default: false)
   - output_type: Output type (0=JSON, 1=TXT, default: 0)
   - yolo_model_type: YOLO model type (n/s/m/l, default: n)
   Response: 
   - If Accept: application/json: JSON with status and output path
   - Otherwise: ZIP file containing results

3. Web Processing API (Queued)
   Endpoint: /web/predict
   Method: POST
   Description: Asynchronously processes files with progress tracking
   Parameters: Same as Direct API
   Response: 
   {
     "task_id": "uuid",
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
     "log": "current_stage",        # Current stage or status message
     "has_detections": boolean,     # Whether any detections were found
     "download_token": "token",     # Only included when task is completed
     "error": "error_message"       # Only included when task has failed
   }

5. Download Results
   Endpoint: /download/<token>
   Method: GET
   Description: Download processed results using token
   Response: ZIP file containing results
   Headers:
   - Content-Type: application/zip
   - Content-Disposition: attachment; filename=result.zip
   - Content-Length: file size in bytes

6. Cancel Task
   Endpoint: /web/cancel/<task_id>
   Method: POST
   Description: Cancel a running or queued task
   Response:
   {
     "status": "success",
     "message": "Task cancelled successfully"
   }
   Error Response:
   {
     "error": "Task not found or already completed"
   }

7. Server Status
   Endpoint: /server-status
   Method: GET
   Description: Get current server status and statistics
   Response:
   {
     "total_tasks_processed": int,      # Total number of tasks processed
     "total_files_processed": int,      # Total number of files processed
     "failed_tasks": int,               # Number of failed tasks
     "cancelled_tasks": int,            # Number of cancelled tasks
     "current_tasks": int,              # Number of currently processing tasks
     "queued_tasks": int,               # Number of tasks in queue
     "uptime_seconds": float,           # Server uptime in seconds
     "max_concurrent_tasks": int,       # Maximum allowed concurrent tasks
     "max_queue_size": int,             # Maximum queue size
     "memory_usage_mb": float,          # Current memory usage in MB
     "cpu_usage_percent": float         # Current CPU usage percentage
   }

Notes:
- All uploads must be ZIP files
- Input files are automatically cleaned up after 2 hours
- Output ZIP files are kept for 4 hours
- Progress updates are provided in real-time
- Failed tasks include error messages in status response
- Cancelled tasks can't be resumed
- Server implements CORS and allows all origins
- All endpoints support OPTIONS method for CORS preflight requests
"""

from flask import Flask, request, send_file
from flask_cors import CORS
from main import execute
import threading
import os
from datetime import datetime
import zipfile
import json
import traceback
import atexit

from utils.api.task_handler import TaskHandler
from utils.api.auth_handler import AuthHandler
from utils.api.file_handler import FileHandler
from utils.api.request_handler import RequestHandler
from utils.api.logger_handler import LoggerHandler

# Global flag to track if background threads are running
background_threads_started = False
background_threads = []

def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)
    
    # Configure maximum file size (e.g., 5GB)
    app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024 * 1024  # 5GB in bytes
    
    # Enable CORS for all origins
    CORS(app, resources={
        r"/*": {
            "origins":  ["https://sightlinks.org/", "*"],  # Allow all origins
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
            "cors_allowed_origins": ["https://sightlinks.org/", "*"],
            "supports_credentials": True,
            "max_age": 86400  # Cache preflight requests for 24 hours
        }
    })
    
    return app

app = create_app()

# Initialize handlers as global variables
file_handler = FileHandler()
auth_handler = AuthHandler()
task_handler = TaskHandler()
request_handler = RequestHandler(file_handler)
logger_handler = LoggerHandler()

def start_background_threads():
    """Start background threads if not already running."""
    global background_threads_started
    
    if not background_threads_started:
        logger_handler.log_system('Starting background threads')
        
        # Start cleanup thread
        cleanup_thread = threading.Thread(
            target=task_handler.cleanup_old_files,
            daemon=True,
            name='cleanup_thread'
        )
        cleanup_thread.start()
        background_threads.append(cleanup_thread)
        
        # Start queue processor thread
        queue_thread = threading.Thread(
            target=task_handler.queue_processor,
            args=(execute,),
            daemon=True,
            name='queue_thread'
        )
        queue_thread.start()
        background_threads.append(queue_thread)
        
        background_threads_started = True
        logger_handler.log_system('Background threads started')

def shutdown_threads():
    """Shutdown background threads gracefully."""
    global background_threads_started
    
    if background_threads_started:
        logger_handler.log_system('Shutting down background threads')
        
        # Signal queue processor to stop
        task_handler.task_queue.put(None)
        
        # Wait for threads to finish
        for thread in background_threads:
            if thread.is_alive():
                thread.join(timeout=5.0)
        
        background_threads_started = False
        logger_handler.log_system('Background threads stopped')

# Register shutdown function
atexit.register(shutdown_threads)

@app.before_first_request
def before_first_request():
    """Initialize the application before the first request."""
    start_background_threads()

@app.route('/test', methods=['GET', 'POST'])
def test_api():
    """Test endpoint to verify API functionality"""
    try:
        # Log the request
        logger_handler.log_request(request.method, '/test')
        
        # Basic server info
        server_info = {
            "status": "operational",
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat(),
            "endpoints": {
                "test": "/test",
                "predict": "/predict",
                "web_predict": "/web/predict",
                "status": "/web/status/<task_id>",
                "download": "/download/<token>"
            }
        }
        
        if request.method == 'POST':
            # Echo back any JSON data sent
            data = request.get_json(silent=True) or {}
            server_info["echo"] = data
            
            # Test file upload if present
            if request.files:
                files = request.files.getlist('file')
                server_info["files"] = [
                    {
                        "filename": f.filename,
                        "content_type": f.content_type,
                        "size": len(f.read()) if f else 0
                    } for f in files
                ]
        
        # Test model paths
        model_checks = {
            "yolo_n": os.path.exists("models/yolo-n.pt"),
            "yolo_s": os.path.exists("models/yolo-s.pt"),
            "mobilenet": os.path.exists("models/MobileNetV3_state_dict_big_train.pth"),
            "vgg16": os.path.exists("models/VGG16_Full_State_Dict.pth")
        }
        server_info["models"] = model_checks
        
        # Test directory structure
        dir_checks = {
            "run_output": os.path.exists("run/output"),
            "run_extract": os.path.exists("run/extract"),
            "input": os.path.exists("input"),
            "models": os.path.exists("models")
        }
        server_info["directories"] = dir_checks
        
        # Test CUDA availability
        try:
            import torch
            server_info["cuda"] = {
                "available": torch.cuda.is_available(),
                "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            }
        except Exception as e:
            server_info["cuda"] = {"error": str(e)}
        
        # System info
        import psutil
        server_info["system"] = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent
        }
        
        return request_handler.create_success_response(server_info)
        
    except Exception as e:
        logger_handler.log_error(str(e), details=traceback.format_exc())
        return request_handler.create_error_response(str(e), 500)


@app.route('/predict', methods=['POST'])
def predict_api():
    """Direct API endpoint that waits for completion"""
    try:
        logger_handler.log_request('POST', '/predict')
        
        # Parse request
        files, params = request_handler.parse_request_parameters(request)
        logger_handler.log_request('POST', '/predict', params=params)
        
        # Create session folders
        session_id, input_folder = file_handler.create_session_folders()
        logger_handler.log_file_operation('CREATE_SESSION', input_folder)
        
        # Save files
        filepaths = request_handler.save_uploaded_files(files, input_folder)
        for filepath in filepaths:
            logger_handler.log_file_operation('SAVE', filepath)
        
        # Process directly
        task_id = task_handler.process_task(None, input_folder, params, execute)
        
        # Get task result
        task = task_handler.get_task_status(task_id)
        if task is None:
            error_msg = f"Task {task_id} not found or failed to process"
            logger_handler.log_error(error_msg)
            return request_handler.create_error_response(error_msg, 500)
        
        logger_handler.log_task_status(task_id, task.get('status', 'unknown'), error=task.get('error'))
        
        if task.get('status') == 'failed':
            error_msg = task.get('error', 'Unknown error occurred')
            logger_handler.log_error(error_msg)
            return request_handler.create_error_response(error_msg, 500)
        
        if not task.get('zip_path'):
            error_msg = "No output file was generated"
            logger_handler.log_error(error_msg)
            return request_handler.create_error_response(error_msg, 500)
        
        logger_handler.log_file_operation('SEND', task['zip_path'])
        # Return response based on client preference
        if request_handler.wants_json_response(request):
            return request_handler.create_success_response({
                'status': 'success',
                'message': 'Processing completed',
                'output_path': task['zip_path']
            })
        else:
            return file_handler.send_file_response(task['zip_path'])
            
    except ValueError as ve:
        logger_handler.log_error(str(ve))
        return request_handler.create_error_response(str(ve), 400)
    except Exception as e:
        logger_handler.log_error(str(e), details=traceback.format_exc())
        return request_handler.create_error_response(str(e), 500)

@app.route('/web/predict', methods=['POST'])
def predict_web():
    """Web endpoint with queuing and progress tracking"""
    try:
        logger_handler.log_request('POST', '/web/predict')
        
        # Parse request
        files, params = request_handler.parse_request_parameters(request)
        logger_handler.log_request('POST', '/web/predict', params=params)
        
        # Check queue size
        if task_handler.task_queue.full():
            logger_handler.log_error('Server busy', details='Queue is full')
            return request_handler.create_error_response('Server is busy. Please try again later.', 503)
        
        # Create session folders
        session_id, input_folder = file_handler.create_session_folders()
        logger_handler.log_file_operation('CREATE_SESSION', input_folder)
        
        # Save files
        filepaths = request_handler.save_uploaded_files(files, input_folder)
        for filepath in filepaths:
            logger_handler.log_file_operation('SAVE', filepath)
        
        # Create task
        task_data = {
            'status': 'queued',
            'progress': 0,
            'stage': 'Queued',
            'created_at': datetime.now(),
            'session_id': session_id,
            'input_folder': input_folder
        }
        task_id = task_handler.add_task(task_data)
        logger_handler.log_task_status(task_id, 'queued', progress=0, stage='Queued')
        
        # Queue task
        task_handler.queue_task({
            'id': task_id,
            'input_folder': input_folder,
            'params': params
        })
        
        return request_handler.create_success_response({
            'task_id': task_id,
            'message': 'Task queued successfully'
        })
        
    except ValueError as ve:
        logger_handler.log_error(str(ve))
        return request_handler.create_error_response(str(ve), 400)
    except Exception as e:
        logger_handler.log_error(str(e), details=traceback.format_exc())
        return request_handler.create_error_response(str(e), 500)


@app.route('/web/status/<task_id>', methods=['GET'])
def get_task_status(task_id):
    """Get task status and progress."""
    try:
        task = task_handler.get_task_status(task_id)
        if not task:
            return request_handler.create_error_response('Task not found', 404)
        
        response = {
            'percentage': task.get('progress', 0),
            'log': task.get('stage', 'Unknown'),
            'has_detections': False  # Default to False
        }
        
        # Add error if task failed
        if task.get('status') == 'failed':
            response['error'] = task.get('error', 'Unknown error')
            return request_handler.create_success_response(response)
        
        # Add error if task cancelled
        if task.get('status') == 'cancelled' or task.get('is_cancelled', False):
            response['error'] = 'Task cancelled by user'
            return request_handler.create_success_response(response)
        
        # Add download token if task completed
        if task.get('status') == 'completed' and task.get('zip_path'):
            # Check if there are any detections
            zip_path = task.get('zip_path')
            if zip_path and os.path.exists(zip_path):
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zf:
                        for filename in zf.namelist():
                            if filename.endswith('output.json'):
                                with zf.open(filename) as f:
                                    content = f.read()
                                    if content:
                                        data = json.loads(content)
                                        response['has_detections'] = any(
                                            len(entry.get('coordinates', [])) > 0 
                                            for entry in data
                                        )
                                        break
                except Exception as e:
                    logger_handler.log_error(f'Error checking detections: {str(e)}')
            
            # Generate download token
            if task.get('session_id'):
                token = auth_handler.generate_download_token(task['session_id'], task_id)
                response['download_token'] = token
            else:
                logger_handler.log_error(f'Missing session_id for completed task: {task_id}')
        
        return request_handler.create_success_response(response)
        
    except Exception as e:
        logger_handler.log_error(str(e), details=traceback.format_exc())
        return request_handler.create_error_response(str(e), 500)

@app.route('/download/<token>', methods=['GET'])
def download_result(token):
    """Download the result file using a valid token."""
    try:
        logger_handler.log_request('GET', f'/download/{token}')
        
        # Verify token
        payload = auth_handler.verify_download_token(token)
        if not payload:
            logger_handler.log_error('Invalid download token')
            return request_handler.create_error_response('Invalid token', 401)
        
        task_id = payload.get('task_id')
        if not task_id:
            logger_handler.log_error('Invalid token payload - missing task_id')
            return request_handler.create_error_response('Invalid token payload', 401)
        
        task = task_handler.get_task_status(task_id)
        if not task:
            logger_handler.log_error(f'Task not found for download: {task_id}')
            return request_handler.create_error_response('Task not found', 404)
        
        # Get and verify zip path
        zip_path = task.get('zip_path')
        if not zip_path:
            logger_handler.log_error(f'ZIP path not found for task: {task_id}')
            return request_handler.create_error_response('ZIP path not found in task data', 404)
        
        if not os.path.exists(zip_path):
            logger_handler.log_error(f'ZIP file not found at path: {zip_path}')
            return request_handler.create_error_response(f'ZIP file not found at path: {zip_path}', 404)
        
        file_size = os.path.getsize(zip_path)
        if file_size == 0:
            logger_handler.log_error(f'Empty ZIP file: {zip_path}')
            return request_handler.create_error_response('ZIP file is empty', 404)
        
        logger_handler.log_file_operation('VERIFY', zip_path)
        
        # Verify zip file integrity
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                contents = zf.namelist()
                if zf.testzip() is not None:
                    logger_handler.log_error(f'Corrupted ZIP file: {zip_path}')
                    return request_handler.create_error_response('ZIP file is corrupted', 500)
                if not contents:
                    logger_handler.log_error(f'Empty ZIP file contents: {zip_path}')
                    return request_handler.create_error_response('ZIP file has no contents', 500)
                logger_handler.log_file_operation('ZIP_CONTENTS', zip_path, details=f"Files: {contents}")
        except Exception as e:
            logger_handler.log_error(f'ZIP verification failed: {str(e)}')
            return request_handler.create_error_response(f'Error verifying ZIP file: {str(e)}', 500)
        
        logger_handler.log_file_operation('DOWNLOAD', zip_path)
        
        # Send file
        try:
            response = send_file(
                zip_path,
                as_attachment=True,
                download_name='result.zip',
                mimetype='application/zip'
            )
            response.headers['Content-Length'] = file_size
            response.headers['Content-Type'] = 'application/zip'
            response.headers['Content-Disposition'] = 'attachment; filename=result.zip'
            logger_handler.log_system(f'File download initiated: {zip_path}')
            return response
        except Exception as e:
            logger_handler.log_error(f'File send failed: {str(e)}')
            return request_handler.create_error_response(f'Error sending file: {str(e)}', 500)
            
    except Exception as e:
        logger_handler.log_error(str(e), details=traceback.format_exc())
        return request_handler.create_error_response(str(e), 500)

@app.route('/web/cancel/<task_id>', methods=['POST'])
def cancel_task(task_id):
    """Cancel a running or queued task."""
    try:
        # Try to cancel the task
        cancelled = task_handler.cancel_task(task_id)
        if cancelled:
            return request_handler.create_success_response({
                'status': 'success',
                'message': 'Task cancelled successfully'
            })
        else:
            return request_handler.create_error_response('Task not found or cannot be cancelled', 404)
            
    except Exception as e:
        logger_handler.log_error(str(e), details=traceback.format_exc())
        return request_handler.create_error_response(str(e), 500)

@app.route('/server-status', methods=['GET'])
def get_server_status():
    """Get current server status and statistics."""
    try:
        logger_handler.log_request('GET', '/server-status')
        status = task_handler.get_server_status()
        return request_handler.create_success_response(status)
    except Exception as e:
        logger_handler.log_error(str(e), details=traceback.format_exc())
        return request_handler.create_error_response(str(e), 500)

if __name__ == '__main__':
    logger_handler.log_system('Starting Flask server on port 8000')
    app.run(host='127.0.0.1', port=8000, debug=False) 
