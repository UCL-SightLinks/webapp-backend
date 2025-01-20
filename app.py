"""
API Documentation

Base URL: http://localhost:8000

1. Direct Processing API
   Endpoint: /api/predict
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
   Response: ZIP file containing results

2. Web Processing API (Queued)
   Endpoint: /web/predict
   Method: POST
   Description: Asynchronously processes files with progress tracking
   Parameters: Same as Direct API
   Response: 
   {
     "task_id": "uuid",
     "message": "Task queued successfully"
   }

3. Task Status
   Endpoint: /web/status/<task_id>
   Method: GET
   Description: Get task status and progress
   Response:
   {
     "percentage": 0-100,           # Progress percentage
     "log": "current_stage",        # Current stage or status message
     "download_token": "token",     # Only included when task is completed
     "error": "error_message"       # Only included when task has failed
   }

4. Download Results
   Endpoint: /download/<token>
   Method: GET
   Description: Download processed results using token
   Response: ZIP file containing results

5. Cancel Task
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

6. Server Status
   Endpoint: /api/server-status
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

from utils.api.task_handler import TaskHandler
from utils.api.auth_handler import AuthHandler
from utils.api.file_handler import FileHandler
from utils.api.request_handler import RequestHandler
from utils.api.logger_handler import LoggerHandler

app = Flask(__name__)
# Enable CORS for all origins
CORS(app, resources={
    r"/*": {
        "origins": "*",  # Allow all origins
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "expose_headers": ["Content-Disposition"]
    }
})

# Initialize handlers
file_handler = FileHandler()
auth_handler = AuthHandler()
task_handler = TaskHandler()
request_handler = RequestHandler(file_handler)
logger_handler = LoggerHandler()

@app.route('/api/predict', methods=['POST'])
def predict_api():
    """Direct API endpoint that waits for completion"""
    try:
        logger_handler.log_request('POST', '/api/predict')
        
        # Parse request
        files, params = request_handler.parse_request_parameters(request)
        logger_handler.log_request('POST', '/api/predict', params=params)
        
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
        logger_handler.log_task_status(task_id, task['status'], error=task.get('error'))
        
        if task['status'] == 'failed':
            logger_handler.log_error(task['error'])
            return request_handler.create_error_response(task['error'], 500)
        
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
        logger_handler.log_request('GET', f'/web/status/{task_id}')
        
        task = task_handler.get_task_status(task_id)
        if not task:
            logger_handler.log_error(f'Task not found: {task_id}')
            return request_handler.create_error_response('Task not found', 404)
        
        response = {
            'percentage': task.get('progress', 0),
            'log': task.get('stage', 'Unknown'),
            'has_detections': False  # Default to False
        }
        
        # Add error if task failed
        if task.get('status') == 'failed':
            response['error'] = task.get('error', 'Unknown error')
            logger_handler.log_task_status(task_id, 'failed', error=response['error'])
            return request_handler.create_success_response(response)
        
        # Add download token if task completed
        if task.get('status') == 'completed':
            # Check if there are any detections
            zip_path = task.get('zip_path')
            if zip_path and os.path.exists(zip_path):
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zf:
                        # Look for output.json in the ZIP
                        for filename in zf.namelist():
                            if filename.endswith('output.json'):
                                with zf.open(filename) as f:
                                    content = f.read()
                                    if content:
                                        data = json.loads(content)
                                        # Check if any entry has detections
                                        response['has_detections'] = any(
                                            len(entry.get('coordinates', [])) > 0 
                                            for entry in data
                                        )
                                        break
                except Exception as e:
                    logger_handler.log_error(f'Error checking detections: {str(e)}')
            
            token = auth_handler.generate_download_token(task['session_id'], task_id)
            response['download_token'] = token
            logger_handler.log_task_status(task_id, 'completed', stage='Processing completed')
        
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
    """Cancel a running or queued task"""
    try:
        logger_handler.log_request('POST', f'/web/cancel/{task_id}')
        
        # Try to cancel the task
        cancelled = task_handler.cancel_task(task_id)
        if cancelled:
            logger_handler.log_task_status(task_id, 'cancelled', stage='Cancelled by user')
            return request_handler.create_success_response({
                'status': 'success',
                'message': 'Task cancelled successfully'
            })
        else:
            logger_handler.log_error(f'Task cancel failed: {task_id}')
            return request_handler.create_error_response('Task not found or already completed', 404)
            
    except Exception as e:
        logger_handler.log_error(str(e), details=traceback.format_exc())
        return request_handler.create_error_response(str(e), 500)

@app.route('/api/server-status', methods=['GET'])
def get_server_status():
    """Get current server status and statistics."""
    try:
        logger_handler.log_request('GET', '/api/server-status')
        status = task_handler.get_server_status()
        return request_handler.create_success_response(status)
    except Exception as e:
        logger_handler.log_error(str(e), details=traceback.format_exc())
        return request_handler.create_error_response(str(e), 500)

# Start background threads
logger_handler.log_system('Starting background threads')
cleanup_thread = threading.Thread(target=task_handler.cleanup_old_files, daemon=True)
cleanup_thread.start()

queue_thread = threading.Thread(target=task_handler.queue_processor, args=(execute,), daemon=True)
queue_thread.start()

logger_handler.log_system('Background threads started')

if __name__ == '__main__':
    logger_handler.log_system('Starting Flask server on port 8000')
    app.run(host='0.0.0.0', port=8000, debug=False) 