"""
API Documentation

Base URL: http://localhost:5010

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
     "status": "queued|processing|completed|failed",
     "progress": 0-100,
     "stage": "current_stage",
     "download_token": "token" (when completed),
     "error": "error_message" (if failed)
   }

4. Download Results
   Endpoint: /download/<token>
   Method: GET
   Description: Download processed results using token
   Response: ZIP file containing results

Notes:
- All uploads must be ZIP files
- Files are automatically cleaned up after 2 hours
- ZIP files are kept for 4 hours
- Progress updates are provided in real-time
- Failed tasks include error messages
"""

from flask import Flask, request, send_file
from flask_cors import CORS
from main import execute
import threading
import os
from datetime import datetime
import zipfile

from utils.api.task_handler import TaskHandler
from utils.api.auth_handler import AuthHandler
from utils.api.file_handler import FileHandler
from utils.api.request_handler import RequestHandler

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

@app.route('/api/predict', methods=['POST'])
def predict_api():
    """Direct API endpoint that waits for completion"""
    try:
        # Parse request
        file, params = request_handler.parse_request_parameters(request)
        
        # Create session folders
        session_id, input_folder = file_handler.create_session_folders()
        
        # Save file
        filepath = request_handler.save_uploaded_file(file, input_folder)
        
        # Process directly
        task_id = task_handler.process_task(None, input_folder, params, execute)
        
        # Get task result
        task = task_handler.get_task_status(task_id)
        
        if task['status'] == 'failed':
            return request_handler.create_error_response(task['error'], 500)
            
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
        return request_handler.create_error_response(str(ve), 400)
    except Exception as e:
        return request_handler.create_error_response(str(e), 500)

@app.route('/web/predict', methods=['POST'])
def predict_web():
    """Web endpoint with queuing and progress tracking"""
    try:
        # Parse request
        file, params = request_handler.parse_request_parameters(request)
        
        # Check queue size
        if task_handler.task_queue.full():
            return request_handler.create_error_response('Server is busy. Please try again later.', 503)
        
        # Create session folders
        session_id, input_folder = file_handler.create_session_folders()
        
        # Save file
        filepath = request_handler.save_uploaded_file(file, input_folder)
        
        # Create task
        task_id = task_handler.add_task({
            'status': 'queued',
            'progress': 0,
            'stage': 'Queued',
            'created_at': datetime.now(),
            'session_id': session_id,
            'input_folder': input_folder
        })
        
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
        print(f"Validation error in predict_web: {str(ve)}")
        return request_handler.create_error_response(str(ve), 400)
    except Exception as e:
        print(f"Error in predict_web: {str(e)}")
        import traceback
        print("Traceback:", traceback.format_exc())
        return request_handler.create_error_response(str(e), 500)

@app.route('/web/status/<task_id>', methods=['GET'])
def task_status(task_id):
    """Get task status and progress"""
    task = task_handler.get_task_status(task_id)
    if not task:
        return request_handler.create_error_response('Task not found', 404)
        
    status_data = {
        'status': task['status'],
        'progress': task['progress'],
        'stage': task.get('stage', '')
    }
    
    if task['status'] == 'completed':
        if 'zip_path' in task:
            # Get session_id from task data or use task_id as fallback
            session_id = task.get('session_id', os.path.basename(os.path.dirname(task['zip_path'])))
            token = auth_handler.generate_download_token(session_id, task_id)
            status_data['download_token'] = token
            print(f"\n=== Token Generation ===")
            print(f"Task ID: {task_id}")
            print(f"Session ID: {session_id}")
            print(f"ZIP Path: {task['zip_path']}")
            print(f"Token Generated: {token}")
        else:
            status_data['error'] = 'ZIP file not found'
            status_data['status'] = 'failed'
            
    elif task['status'] == 'failed':
        status_data['error'] = task.get('error', 'Unknown error')
        
    return request_handler.create_success_response(status_data)

@app.route('/download/<token>', methods=['GET'])
def download_result(token):
    """Download the result file using a valid token."""
    try:
        # Verify token
        payload = auth_handler.verify_download_token(token)
        if not payload:
            return request_handler.create_error_response('Invalid token', 401)
            
        # Get task status to find the zip file path
        task_id = payload.get('task_id')
        if not task_id:
            return request_handler.create_error_response('Invalid token payload', 401)
            
        task = task_handler.get_task_status(task_id)
        if not task:
            return request_handler.create_error_response('Task not found', 404)
            
        print(f"\n=== Download Request Details ===")
        print(f"Task ID: {task_id}")
        print(f"Task Status: {task.get('status')}")
        print(f"Task Data: {task}")
        
        # Get and verify zip path
        zip_path = task.get('zip_path')
        if not zip_path:
            return request_handler.create_error_response('ZIP path not found in task data', 404)
            
        if not os.path.exists(zip_path):
            return request_handler.create_error_response(f'ZIP file not found at path: {zip_path}', 404)
            
        file_size = os.path.getsize(zip_path)
        if file_size == 0:
            return request_handler.create_error_response('ZIP file is empty', 404)
            
        print(f"ZIP Path: {zip_path}")
        print(f"File exists: Yes")
        print(f"File size: {file_size} bytes")
        
        # Verify zip file integrity
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                contents = zf.namelist()
                print(f"ZIP contents: {contents}")
                
                # Get detailed file info
                for name in contents:
                    info = zf.getinfo(name)
                    print(f"File: {name}")
                    print(f"  Size: {info.file_size} bytes")
                    print(f"  Compressed: {info.compress_size} bytes")
                
                if zf.testzip() is not None:
                    return request_handler.create_error_response('ZIP file is corrupted', 500)
                    
                if not contents:
                    return request_handler.create_error_response('ZIP file has no contents', 500)
        except zipfile.BadZipFile as e:
            return request_handler.create_error_response(f'Invalid ZIP file: {str(e)}', 500)
        except Exception as e:
            return request_handler.create_error_response(f'Error verifying ZIP file: {str(e)}', 500)
        
        print("\n=== Sending File ===")
        print(f"File: {zip_path}")
        print(f"Size: {file_size} bytes")
        print(f"Contents: {contents}")
        
        # Always send file for download endpoint
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
            print("File sending initiated successfully")
            return response
        except Exception as e:
            print(f"Error sending file: {str(e)}")
            return request_handler.create_error_response(f'Error sending file: {str(e)}', 500)
            
    except Exception as e:
        print(f"Download error: {str(e)}")
        print("Traceback:", traceback.format_exc())
        return request_handler.create_error_response(str(e), 500)

# Start background threads
cleanup_thread = threading.Thread(target=task_handler.cleanup_old_files, daemon=True)
cleanup_thread.start()

queue_thread = threading.Thread(target=task_handler.queue_processor, args=(execute,), daemon=True)
queue_thread.start()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5010) 