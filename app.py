from flask import Flask, request, jsonify, send_file
from main import execute
import os
from werkzeug.utils import secure_filename
import shutil
import uuid
from datetime import datetime, timedelta
import jwt
import threading
import time
import queue
from utils.compress import compress_folder_to_zip

app = Flask(__name__)

# Configure base folders and settings
BASE_UPLOAD_FOLDER = 'input'
BASE_OUTPUT_FOLDER = 'run/output'
ALLOWED_EXTENSIONS = {'zip'}
JWT_SECRET = os.getenv('JWT_SECRET', 'your-secret-key')  # In production, use environment variable
MAX_FILE_AGE_HOURS = 2
MAX_QUEUE_SIZE = 10

# Task queue and tracking
task_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
active_tasks = {}
task_lock = threading.Lock()

# Create base directories if they don't exist
for folder in [BASE_UPLOAD_FOLDER, BASE_OUTPUT_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_session_folders():
    """Create unique session folders for input and output"""
    session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
    input_folder = os.path.join(BASE_UPLOAD_FOLDER, session_id)
    os.makedirs(input_folder, exist_ok=True)
    return session_id, input_folder

def generate_download_token(session_id, output_folder):
    """Generate a JWT token for file download"""
    expiration = datetime.utcnow() + timedelta(hours=MAX_FILE_AGE_HOURS)
    token = jwt.encode({
        'session_id': session_id,
        'output_folder': output_folder,
        'exp': expiration
    }, JWT_SECRET, algorithm='HS256')
    return token

def verify_download_token(token):
    """Verify JWT token and return payload if valid"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
        return payload
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None

def process_task(task_id, input_folder, params):
    """Process a single task and update its status"""
    try:
        with task_lock:
            active_tasks[task_id]['status'] = 'processing'
            active_tasks[task_id]['progress'] = 10
            print(f"Processing task {task_id}: Starting...")

        # Execute the model - unpack params as individual arguments
        with task_lock:
            active_tasks[task_id]['progress'] = 20
            print(f"Processing task {task_id}: Executing model...")

        try:
            execute(
                input_folder,  # uploadDir
                int(params['input_type']),  # inputType
                float(params['classification_threshold']),  # classificationThreshold
                float(params['prediction_threshold']),  # predictionThreshold
                params['save_labeled_image'].lower() == 'true',  # saveLabeledImage
                int(params['output_type']),  # outputType
                params['yolo_model_type']  # yoloModelType
            )
            # Update progress after model execution
            with task_lock:
                active_tasks[task_id]['progress'] = 80
                print(f"Processing task {task_id}: Model execution complete")
        except Exception as e:
            print(f"Error during model execution: {str(e)}")
            raise

        # Get the output folder path based on timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_folder = os.path.join("run/output", timestamp)
        
        # Verify output folder exists
        if not os.path.exists(output_folder):
            # Try finding the most recent output folder
            output_folders = [f for f in os.listdir("run/output") if os.path.isdir(os.path.join("run/output", f))]
            if output_folders:
                output_folders.sort(reverse=True)  # Sort by name (timestamp) in reverse order
                output_folder = os.path.join("run/output", output_folders[0])
                print(f"Processing task {task_id}: Found output folder {output_folder}")

        if not os.path.exists(output_folder):
            raise Exception("Output folder not found")

        with task_lock:
            active_tasks[task_id]['progress'] = 90
            print(f"Processing task {task_id}: Creating ZIP file...")

        # Create ZIP file for output
        zip_name = f"{os.path.basename(output_folder)}.zip"
        zip_path = compress_folder_to_zip(output_folder, zip_name)
        
        if isinstance(zip_path, str) and os.path.exists(zip_path):
            with task_lock:
                active_tasks[task_id]['status'] = 'completed'
                active_tasks[task_id]['progress'] = 100
                active_tasks[task_id]['output_folder'] = output_folder
                active_tasks[task_id]['zip_path'] = zip_path
                print(f"Processing task {task_id}: Completed successfully")
        else:
            raise Exception(f"Failed to create ZIP file: {zip_path}")

    except Exception as e:
        error_msg = str(e)
        print(f"Error processing task {task_id}: {error_msg}")
        with task_lock:
            active_tasks[task_id]['status'] = 'failed'
            active_tasks[task_id]['error'] = error_msg
            # Clean up input folder on error
            if os.path.exists(input_folder):
                try:
                    shutil.rmtree(input_folder)
                except Exception as cleanup_error:
                    print(f"Error cleaning up input folder: {str(cleanup_error)}")
        raise

def queue_processor():
    """Process tasks from the queue"""
    while True:
        try:
            task = task_queue.get()
            if task is None:  # Shutdown signal
                break
            
            process_task(task['id'], task['input_folder'], task['params'])
            task_queue.task_done()
        except Exception as e:
            print(f"Task processing error: {str(e)}")

def cleanup_old_files():
    """Clean up files older than MAX_FILE_AGE_HOURS"""
    while True:
        try:
            current_time = datetime.now()
            # Clean up output folders and their ZIPs
            for folder in os.listdir(BASE_OUTPUT_FOLDER):
                folder_path = os.path.join(BASE_OUTPUT_FOLDER, folder)
                if os.path.isdir(folder_path):
                    folder_time = datetime.strptime(folder.split('_')[0], '%Y%m%d')
                    if current_time - folder_time > timedelta(hours=MAX_FILE_AGE_HOURS):
                        # Remove ZIP file if exists
                        zip_path = os.path.join(os.path.dirname(folder_path), f"{folder}.zip")
                        if os.path.exists(zip_path):
                            os.remove(zip_path)
                        shutil.rmtree(folder_path)
            
            # Clean up input folders
            for folder in os.listdir(BASE_UPLOAD_FOLDER):
                folder_path = os.path.join(BASE_UPLOAD_FOLDER, folder)
                if os.path.isdir(folder_path):
                    folder_time = datetime.strptime(folder.split('_')[0], '%Y%m%d')
                    if current_time - folder_time > timedelta(hours=MAX_FILE_AGE_HOURS):
                        shutil.rmtree(folder_path)
        except Exception as e:
            print(f"Cleanup error: {str(e)}")
        time.sleep(1800)  # Run every 30 minutes

@app.route('/api/predict', methods=['POST'])
def predict_api():
    """Direct API endpoint that waits for completion"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400

    try:
        # Create session folders
        session_id, input_folder = create_session_folders()

        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(input_folder, filename)
        file.save(filepath)

        # Get parameters and convert types
        params = {
            'input_type': request.form.get('input_type', '0'),
            'classification_threshold': request.form.get('classification_threshold', '0.35'),
            'prediction_threshold': request.form.get('prediction_threshold', '0.5'),
            'save_labeled_image': request.form.get('save_labeled_image', 'false'),
            'output_type': request.form.get('output_type', '0'),
            'yolo_model_type': request.form.get('yolo_model_type', 'n')
        }

        # Process directly
        execute(
            input_folder,  # uploadDir
            int(params['input_type']),  # inputType
            float(params['classification_threshold']),  # classificationThreshold
            float(params['prediction_threshold']),  # predictionThreshold
            params['save_labeled_image'].lower() == 'true',  # saveLabeledImage
            int(params['output_type']),  # outputType
            params['yolo_model_type']  # yoloModelType
        )

        # Get the output folder path based on timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_folder = os.path.join("run/output", timestamp)
        
        # Verify output folder exists
        if not os.path.exists(output_folder):
            # Try finding the most recent output folder
            output_folders = [f for f in os.listdir("run/output") if os.path.isdir(os.path.join("run/output", f))]
            if output_folders:
                output_folders.sort(reverse=True)  # Sort by name (timestamp) in reverse order
                output_folder = os.path.join("run/output", output_folders[0])

        if not os.path.exists(output_folder):
            raise Exception("Output folder not found")

        # Create ZIP file
        zip_name = f"{os.path.basename(output_folder)}.zip"
        zip_path = compress_folder_to_zip(output_folder, zip_name)

        if isinstance(zip_path, str) and os.path.exists(zip_path):
            return send_file(
                zip_path,
                as_attachment=True,
                attachment_filename='result.zip',
                mimetype='application/zip'
            )
        else:
            raise Exception(f"Failed to create ZIP file: {zip_path}")

    except Exception as e:
        if 'input_folder' in locals() and os.path.exists(input_folder):
            shutil.rmtree(input_folder)
        return jsonify({'error': str(e)}), 500

@app.route('/web/predict', methods=['POST'])
def predict_web():
    """Web endpoint with queuing and progress tracking"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400

    try:
        # Check queue size
        if task_queue.full():
            return jsonify({'error': 'Server is busy. Please try again later.'}), 503

        # Create session folders
        session_id, input_folder = create_session_folders()

        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(input_folder, filename)
        file.save(filepath)

        # Get parameters (keep as strings, convert in process_task)
        params = {
            'input_type': request.form.get('input_type', '0'),
            'classification_threshold': request.form.get('classification_threshold', '0.35'),
            'prediction_threshold': request.form.get('prediction_threshold', '0.5'),
            'save_labeled_image': request.form.get('save_labeled_image', 'false'),
            'output_type': request.form.get('output_type', '0'),
            'yolo_model_type': request.form.get('yolo_model_type', 'n')
        }

        # Create task
        task_id = str(uuid.uuid4())

        # Initialize task status
        with task_lock:
            active_tasks[task_id] = {
                'status': 'queued',
                'progress': 0,
                'created_at': datetime.now(),
                'session_id': session_id,
                'input_folder': input_folder
            }

        # Add to queue
        task_queue.put({
            'id': task_id,
            'input_folder': input_folder,
            'params': params
        })

        return jsonify({
            'task_id': task_id,
            'message': 'Task queued successfully'
        })

    except Exception as e:
        if 'input_folder' in locals() and os.path.exists(input_folder):
            shutil.rmtree(input_folder)
        return jsonify({'error': str(e)}), 500

@app.route('/web/status/<task_id>', methods=['GET'])
def task_status(task_id):
    """Get task status and progress"""
    with task_lock:
        task = active_tasks.get(task_id)
        if not task:
            return jsonify({'error': 'Task not found'}), 404
        
        status_data = {
            'status': task['status'],
            'progress': task['progress']
        }

        if task['status'] == 'completed':
            # Generate download token with the actual output folder
            token = generate_download_token(task['session_id'], task['output_folder'])
            status_data['download_token'] = token
            status_data['output_folder'] = task['output_folder']  # Include for debugging

        elif task['status'] == 'failed':
            status_data['error'] = task.get('error', 'Unknown error')

        return jsonify(status_data)

@app.route('/download/<token>', methods=['GET'])
def download_result(token):
    """Download the result file using a valid token"""
    payload = verify_download_token(token)
    if not payload:
        return jsonify({'error': 'Invalid or expired token'}), 401

    output_folder = payload['output_folder']
    if not output_folder or not os.path.exists(output_folder):
        return jsonify({'error': 'Output files no longer exist'}), 404

    try:
        # Create ZIP file if it doesn't exist
        zip_path = os.path.join(os.path.dirname(output_folder), f"{os.path.basename(output_folder)}.zip")
        if not os.path.exists(zip_path):
            zip_path = compress_folder_to_zip(output_folder, zip_path)

        return send_file(
            zip_path,
            as_attachment=True,
            attachment_filename='result.zip',
            mimetype='application/zip'
        )
    except Exception as e:
        print(f"Error during download: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Start background threads
cleanup_thread = threading.Thread(target=cleanup_old_files, daemon=True)
cleanup_thread.start()

queue_thread = threading.Thread(target=queue_processor, daemon=True)
queue_thread.start()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5010) 