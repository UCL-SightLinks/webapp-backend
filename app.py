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

# Create a wrapper for execute to ensure it only receives expected parameters
def execute_wrapper(uploadDir, inputType, classificationThreshold, predictionThreshold, saveLabeledImage, outputType, yoloModelType):
    return execute(uploadDir, inputType, classificationThreshold, predictionThreshold, saveLabeledImage, outputType, yoloModelType)

# Create a special wrapper for queue_processor to protect execute_func at task level
def queue_processor_wrapper(task_handler, execute_wrapper):
    task_handler.queue_processor(execute_wrapper)

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
            target=queue_processor_wrapper,
            args=(task_handler, execute_wrapper),
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

# Start background threads when the app is created
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
            "yolo_m": os.path.exists("models/yolo-m.pt"),
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
        
        # DEBUG - Log full request details
        print("\n========== DETAILED REQUEST INSPECTION (/predict) ==========")
        print(f"Request method: {request.method}")
        print(f"Content-Type: {request.content_type}")
        print(f"Content-Length: {request.headers.get('Content-Length')}")
        print(f"Form keys: {list(request.form.keys())}")
        print(f"Files keys: {list(request.files.keys())}")
        
        # Inspect each file key
        for key in request.files:
            files_list = request.files.getlist(key)
            print(f"\nFile key '{key}' has {len(files_list)} file(s):")
            for idx, file in enumerate(files_list):
                if file and file.filename:
                    print(f"  [{idx}] Filename: {file.filename}")
                    print(f"      Content-Type: {file.content_type}")
                    file_size = 0
                    file.stream.seek(0, os.SEEK_END)
                    file_size = file.stream.tell()
                    file.stream.seek(0)  # Reset file pointer
                    print(f"      Size: {file_size} bytes")
                else:
                    print(f"  [{idx}] Empty file")
        print("===========================================================\n")
        
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
        task_id = task_handler.process_task(None, input_folder, params, execute_wrapper)
        
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
            # FIXED: Use direct send_file instead of file_handler to avoid double-wrapping
            zip_path = task['zip_path']
            
            # Fixed filename for consistency
            timestamp = datetime.now().strftime("%Y%m%d")
            filename = f"result_{timestamp}.zip"
            
            # Use Flask's send_file directly
            return send_file(
                zip_path,
                mimetype='application/zip',
                as_attachment=True,
                download_name=filename
            )
            
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
        
        # DEBUG - Log full request details
        print("\n========== DETAILED REQUEST INSPECTION (/web/predict) ==========")
        print(f"Request method: {request.method}")
        print(f"Content-Type: {request.content_type}")
        print(f"Content-Length: {request.headers.get('Content-Length')}")
        print(f"Form keys: {list(request.form.keys())}")
        print(f"Files keys: {list(request.files.keys())}")
        
        # Inspect each file key
        for key in request.files:
            files_list = request.files.getlist(key)
            print(f"\nFile key '{key}' has {len(files_list)} file(s):")
            for idx, file in enumerate(files_list):
                if file and file.filename:
                    print(f"  [{idx}] Filename: {file.filename}")
                    print(f"      Content-Type: {file.content_type}")
                    file_size = 0
                    file.stream.seek(0, os.SEEK_END)
                    file_size = file.stream.tell()
                    file.stream.seek(0)  # Reset file pointer
                    print(f"      Size: {file_size} bytes")
                else:
                    print(f"  [{idx}] Empty file")
        print("===========================================================\n")
        
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
        
        # Simple status response: false for not completed, true for completed with download available
        if task.get('status') == 'completed':
            # First check if the task already has the has_detections flag
            if 'has_detections' in task:
                has_detections = task.get('has_detections', False)
                total_detections = 0  # Will be determined if needed
                
                logger_handler.log_system(f'Using stored detection status: {has_detections}')
                
                # If we have no detections, we're done
                if not has_detections:
                    logger_handler.log_system(f'Task {task_id} has no detections (from stored status)')
                    # Generate download token for accessing the no_detections.txt file
                    if task.get('session_id'):
                        token = auth_handler.generate_download_token(task['session_id'], task_id)
                        return request_handler.create_success_response({
                            'completed': True,
                            'download_token': token,
                            'has_detections': False
                        })
                
                # For tasks with detections, we need the download token
                if task.get('session_id') and task.get('zip_path'):
                    token = auth_handler.generate_download_token(task['session_id'], task_id)
                    
                    # Count detections if we need to
                    if has_detections and 'total_detections' not in task:
                        # We'll need to count the detections
                        output_folder = task.get('output_folder')
                        if not output_folder and task.get('zip_path'):
                            output_folder = os.path.dirname(task.get('zip_path'))
                            
                        if output_folder and os.path.exists(output_folder):
                            # Try to count from JSON first
                            json_path = os.path.join(output_folder, "detections.json")
                            if os.path.exists(json_path) and os.path.getsize(json_path) > 10:
                                try:
                                    with open(json_path, 'r') as f:
                                        data = json.load(f)
                                        for item in data:
                                            coordinates = item.get('coordinates', [])
                                            if isinstance(coordinates, list):
                                                total_detections += len(coordinates)
                                        logger_handler.log_system(f'Counted {total_detections} detections from JSON')
                                except Exception as e:
                                    logger_handler.log_error(f'Error reading JSON: {str(e)}')
                    else:
                        # Use already stored count
                        total_detections = task.get('total_detections', 0)
                        
                    response_data = {
                        'completed': True,
                        'download_token': token,
                        'has_detections': has_detections
                    }
                    
                    # Include total_detections if any were found
                    if total_detections > 0:
                        response_data['total_detections'] = total_detections
                    
                    return request_handler.create_success_response(response_data)
            
            # Fall back to old detection checking logic if necessary
            # Check if we have a zip path or just an output folder
            if task.get('zip_path'):
                # Generate download token
                if task.get('session_id'):
                    token = auth_handler.generate_download_token(task['session_id'], task_id)
                    
                    # Determine if the task has detections - improved path handling
                    zip_path = task.get('zip_path', '')
                    zip_directory = os.path.dirname(zip_path)
                    session_id = task.get('session_id', '')
                    
                    # Check parent directory of zip for output folder structure
                    output_folder = None
                    has_detections = False
                    total_detections = 0
                    
                    # Try finding the actual output folder which contains the detection files
                    # The output folder may be named with the session_id or may be in the same directory as the zip
                    logger_handler.log_system(f'Checking for detections in session: {session_id}')
                    
                    # First check for a folder in run/output with the session_id name
                    session_output = os.path.join('run/output', session_id)
                    
                    if os.path.exists(session_output) and os.path.isdir(session_output):
                        output_folder = session_output
                        logger_handler.log_system(f'Found output folder at session path: {output_folder}')
                    else:
                        # Next, look for folders in the zip directory that might contain the detection files
                        possible_folders = []
                        if os.path.exists(zip_directory) and os.path.isdir(zip_directory):
                            for item in os.listdir(zip_directory):
                                item_path = os.path.join(zip_directory, item)
                                if os.path.isdir(item_path):
                                    possible_folders.append(item_path)
                        
                        if len(possible_folders) == 1:
                            # If there's only one folder, use it
                            output_folder = possible_folders[0]
                            logger_handler.log_system(f'Found single output folder: {output_folder}')
                        elif len(possible_folders) > 1:
                            # If there are multiple folders, try to find the one with detection files
                            for folder in possible_folders:
                                if os.path.exists(os.path.join(folder, 'detections.json')):
                                    output_folder = folder
                                    logger_handler.log_system(f'Found output folder with JSON: {output_folder}')
                                    break
                            
                            if not output_folder:
                                # If still not found, use the most recently modified folder
                                possible_folders.sort(key=os.path.getmtime, reverse=True)
                                output_folder = possible_folders[0]
                                logger_handler.log_system(f'Using most recent folder: {output_folder}')
                    
                    # If we still don't have an output folder, just use the zip directory
                    if not output_folder:
                        output_folder = zip_directory
                        logger_handler.log_system(f'Falling back to zip directory: {output_folder}')
                    
                    # Check for the no_detections marker file
                    no_detections_marker = os.path.join(output_folder, "no_detections.txt")
                    if os.path.exists(no_detections_marker):
                        has_detections = False
                        logger_handler.log_system(f'No detections marker file found: {no_detections_marker}')
                    else:
                        # First check for detections.json which is the most reliable source
                        json_path = os.path.join(output_folder, "detections.json")
                        
                        if os.path.exists(json_path) and os.path.getsize(json_path) > 10:
                            try:
                                logger_handler.log_system(f'Found JSON output file: {json_path} (size: {os.path.getsize(json_path)})')
                                with open(json_path, 'r') as f:
                                    data = json.load(f)
                                    # Count detections across all images
                                    for item in data:
                                        coordinates = item.get('coordinates', [])
                                        if isinstance(coordinates, list):
                                            detection_count = len(coordinates)
                                            total_detections += detection_count
                                            image_name = item.get('image', 'unknown')
                                            logger_handler.log_system(f'Image {image_name}: {detection_count} detections')
                                    
                                    # If we found any detections in the JSON, set flag to true
                                    has_detections = total_detections > 0
                                    logger_handler.log_system(f'Found {total_detections} total detections in JSON data')
                            except Exception as e:
                                logger_handler.log_error(f'Error reading JSON: {str(e)}')
                        
                        # If no JSON or no detections found in JSON, check TXT files
                        if not has_detections and os.path.exists(output_folder):
                            txt_files = [f for f in os.listdir(output_folder) if f.endswith('.txt') and f != "no_detections.txt"]
                            
                            if txt_files:
                                logger_handler.log_system(f'Found {len(txt_files)} TXT files in output folder')
                                for txt_file in txt_files:
                                    txt_path = os.path.join(output_folder, txt_file)
                                    # Count lines as detections
                                    if os.path.getsize(txt_path) > 0:
                                        try:
                                            with open(txt_path, 'r') as f:
                                                lines = [line.strip() for line in f if line.strip()]
                                                file_detections = len(lines)
                                                total_detections += file_detections
                                                logger_handler.log_system(f'TXT file {txt_file}: {file_detections} detections')
                                        except Exception as e:
                                            logger_handler.log_error(f'Error reading TXT file: {str(e)}')
                                
                                has_detections = total_detections > 0
                                logger_handler.log_system(f'Found {total_detections} total detections in TXT files')
                    
                    logger_handler.log_system(f'Final detection status: has_detections={has_detections}, total_detections={total_detections}')
                    
                    response_data = {
                        'completed': True,
                        'download_token': token,
                        'has_detections': has_detections
                    }
                    
                    # Include total_detections if any were found
                    if total_detections > 0:
                        response_data['total_detections'] = total_detections
                    
                    return request_handler.create_success_response(response_data)
            # Handle the case where there's just an output folder (no ZIP)
            elif task.get('output_folder'):
                output_folder = task.get('output_folder')
                no_detections_marker = os.path.join(output_folder, "no_detections.txt")
                has_detections = not os.path.exists(no_detections_marker)
                
                # Generate download token for accessing the no_detections.txt file if needed
                if task.get('session_id'):
                    token = auth_handler.generate_download_token(task['session_id'], task_id)
                    return request_handler.create_success_response({
                        'completed': True,
                        'download_token': token,
                        'has_detections': has_detections
                    })
                    
            # Missing session_id case
            error_msg = f'Missing session_id for completed task: {task_id}'
            logger_handler.log_error(error_msg)
            return request_handler.create_error_response(error_msg, 500)
        
        # Check for error state
        if task.get('status') == 'failed':
            error_msg = task.get('error', 'Task processing failed')
            logger_handler.log_error(f'Task {task_id} failed: {error_msg}')
            return request_handler.create_success_response({
                'completed': False,
                'error': True,
                'error_message': error_msg
            })
        
        # For cancelled tasks, add a specific response
        if task.get('status') == 'cancelled':
            logger_handler.log_task_status(task_id, 'cancelled', stage='Checking cancelled task status')
            return request_handler.create_success_response({
                'completed': False,
                'cancelled': True,
                'message': 'Task was cancelled'
            })
        
        # For all other states (queued, processing), return false
        return request_handler.create_success_response({
            'completed': False
        })
        
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
        
        # Get application root directory for absolute paths
        app_root = os.path.dirname(os.path.abspath(__file__))
        
        # Get task status which will try to load from disk if not in memory
        task = task_handler.get_task_status(task_id)
        logger_handler.log_system(f'App root directory: {app_root}')
        logger_handler.log_system(f'Task info: {task}')
        
        # Check if task exists
        if task.get('status') == 'unknown':
            logger_handler.log_error(f'Task not found for download: {task_id}')
            return request_handler.create_error_response('Task not found', 404)
        
        # Check if session_id is available
        session_id = task.get('session_id')
        if not session_id:
            logger_handler.log_error(f'Session ID not found for task: {task_id}')
            return request_handler.create_error_response('Session ID not found', 404)
        
        # If the task has a zip_path, use it directly
        if task.get('zip_path'):
            zip_path = task.get('zip_path')
            
            # Make sure zip_path is an absolute path
            if not os.path.isabs(zip_path):
                zip_path = os.path.join(app_root, zip_path)
                
            logger_handler.log_system(f'Using stored ZIP path: {zip_path}')
            
            if os.path.exists(zip_path) and os.path.getsize(zip_path) > 0:
                # Setup proper headers with a consistent filename
                timestamp = datetime.now().strftime("%Y%m%d")
                filename = f"result_{timestamp}.zip"
                
                logger_handler.log_file_operation('DOWNLOAD', zip_path)
                response = send_file(
                    zip_path,
                    mimetype='application/zip',
                    as_attachment=True,
                    download_name=filename
                )
                
                # Set detection headers
                response.headers['X-Has-Detections'] = str(task.get('has_detections', False)).lower()
                if task.get('total_detections') is not None:
                    response.headers['X-Total-Detections'] = str(task.get('total_detections', 0))
                
                logger_handler.log_system(f'File download initiated with filename: {filename}, detections: {task.get("total_detections", 0)}')
                return response
            else:
                logger_handler.log_error(f'ZIP file not found or empty: {zip_path}')
                
        # If the task has an output_folder, try to find the output folder
        if task.get('output_folder'):
            output_folder = task.get('output_folder')
            
            # Make sure output_folder is an absolute path
            if not os.path.isabs(output_folder):
                output_folder = os.path.join(app_root, output_folder)
                
            logger_handler.log_system(f'Using output folder: {output_folder}')
            
            # Check for no_detections.txt if has_detections is False
            if not task.get('has_detections', True):
                no_detections_marker = os.path.join(output_folder, "no_detections.txt")
                if os.path.exists(no_detections_marker):
                    logger_handler.log_system(f'Sending no_detections marker file for task {task_id}')
                    timestamp = datetime.now().strftime("%Y%m%d")
                    response = send_file(
                        no_detections_marker, 
                        mimetype='text/plain',
                        as_attachment=True,
                        download_name=f'result_{timestamp}.txt'
                    )
                    response.headers['X-Has-Detections'] = 'false'
                    return response
            
            # Look for ZIP files in the output folder's parent directory
            parent_dir = os.path.dirname(output_folder)
            timestamp_prefix = '_'.join(session_id.split('_')[:2])
            
            for filename in os.listdir(parent_dir):
                if filename.startswith('result_') and filename.endswith('.zip'):
                    zip_path = os.path.join(parent_dir, filename)
                    logger_handler.log_system(f'Found ZIP file: {zip_path}')
                    
                    # Update task with zip_path
                    with task_handler.task_lock:
                        if task_id in task_handler.active_tasks:
                            task_handler.active_tasks[task_id]['zip_path'] = zip_path
                            task_handler._save_tasks()
                    
                    # Send the file
                    timestamp = datetime.now().strftime("%Y%m%d")
                    download_filename = f"result_{timestamp}.zip"
                    
                    logger_handler.log_file_operation('DOWNLOAD', zip_path)
                    response = send_file(
                        zip_path,
                        mimetype='application/zip',
                        as_attachment=True,
                        download_name=download_filename
                    )
                    
                    # Set detection headers
                    response.headers['X-Has-Detections'] = str(task.get('has_detections', True)).lower()
                    if task.get('total_detections') is not None:
                        response.headers['X-Total-Detections'] = str(task.get('total_detections', 0))
                    
                    logger_handler.log_system(f'File download initiated with filename: {download_filename}, detections: {task.get("total_detections", 0)}')
                    return response
        
        # If we got here, we couldn't find a valid download file
        logger_handler.log_error(f'ZIP path not found for task: {task_id}')
        return request_handler.create_error_response('ZIP path not found in task data', 404)
        
    except Exception as e:
        logger_handler.log_error(f'Download error: {str(e)}', details=traceback.format_exc())
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

@app.route('/')
def index():
    # your code here
    return 'Hello World'

@app.route('/debug_files', methods=['POST'])
def debug_files():
    """Debug endpoint that just shows received files"""
    print("\n==== DEBUG FILES ENDPOINT ====")
    print(f"Request method: {request.method}")
    print(f"Content-Type: {request.content_type}")
    
    # Print all request files
    received_files = []
    print("Files in request.files.keys():", list(request.files.keys()))
    
    for key in request.files:
        files_list = request.files.getlist(key)
        print(f"Key '{key}' has {len(files_list)} file(s)")
        
        for idx, file in enumerate(files_list):
            if file and file.filename:
                file_info = {
                    "key": key,
                    "index": idx,
                    "filename": file.filename,
                    "content_type": file.content_type
                }
                received_files.append(file_info)
                print(f"  File {idx}: {file.filename} ({file.content_type})")
    
    # Return a simple JSON response with the files received
    return {
        "message": f"Received {len(received_files)} files",
        "files": received_files
    }, 200

if __name__ == '__main__':
    logger_handler.log_system('Starting Flask server on port 8000')
    app.run(host='127.0.0.1', port=8000, debug=False) 
