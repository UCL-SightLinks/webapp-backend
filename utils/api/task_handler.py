"""Task handler module for managing task processing and cleanup."""

import os
import shutil
import queue
import threading
import time
import uuid
from datetime import datetime, timedelta
import traceback
import zipfile
from utils.compress import compress_folder_to_zip
from utils.api.logger_handler import LoggerHandler

class TaskHandler:
    """Handles task processing, queuing, and cleanup."""
    
    def __init__(self):
        """Initialize task handler with configuration."""
        self.BASE_UPLOAD_FOLDER = 'input'
        self.BASE_OUTPUT_FOLDER = 'run/output'
        self.MAX_FILE_AGE_HOURS = 2  # For input files
        self.MAX_OUTPUT_AGE_HOURS = 4  # For output files and zips
        self.MAX_QUEUE_SIZE = 10
        
        # Task queue and tracking
        self.task_queue = queue.Queue(maxsize=self.MAX_QUEUE_SIZE)
        self.active_tasks = {}
        self.task_lock = threading.Lock()
        self.cancelled_tasks = set()  # Track cancelled tasks
        
        # Initialize logger
        self.logger = LoggerHandler()
        
        # Create base directories
        for folder in [self.BASE_UPLOAD_FOLDER, self.BASE_OUTPUT_FOLDER]:
            if not os.path.exists(folder):
                os.makedirs(folder)
                self.logger.log_system(f'Created base directory: {folder}')
    
    def add_task(self, task_data):
        """Add a new task to active tasks."""
        task_id = str(uuid.uuid4())
        with self.task_lock:
            self.active_tasks[task_id] = task_data
            self.logger.log_task_status(task_id, 'created', stage='Task created')
        return task_id
    
    def queue_task(self, task):
        """Add a task to the processing queue."""
        self.task_queue.put(task)
        self.logger.log_task_status(task['id'], 'queued', stage='Added to processing queue')
    
    def get_task_status(self, task_id):
        """Get the status of a task."""
        with self.task_lock:
            return self.active_tasks.get(task_id)
    
    def cancel_task(self, task_id):
        """Cancel a task if it's still running or queued."""
        with self.task_lock:
            task = self.active_tasks.get(task_id)
            if not task:
                self.logger.log_error(f'Task not found for cancellation: {task_id}')
                return False
            
            if task['status'] not in ['queued', 'processing']:
                self.logger.log_error(f'Cannot cancel task {task_id} in status: {task["status"]}')
                return False
            
            self.cancelled_tasks.add(task_id)
            task['status'] = 'cancelled'
            task['stage'] = 'Cancelled by user'
            task['progress'] = 100
            
            input_folder = task.get('input_folder')
            if input_folder and os.path.exists(input_folder):
                try:
                    shutil.rmtree(input_folder)
                    self.logger.log_cleanup('input_folder', input_folder)
                except Exception as e:
                    self.logger.log_error(f'Error cleaning up input folder: {str(e)}')
            
            self.logger.log_task_status(task_id, 'cancelled', progress=100, stage='Cancelled by user')
            return True
    
    def process_task(self, task_id, input_folder, params, execute_func):
        """Process a single task and update its status."""
        output_folder = None
        try:
            if task_id in self.cancelled_tasks:
                self.logger.log_task_status(task_id, 'cancelled', stage='Task was cancelled before processing')
                return None
            
            self.logger.log_task_status(task_id, 'processing', progress=0, stage='Starting processing')
            
            session_id = os.path.basename(input_folder)
            self.logger.log_system(f'Processing task {task_id} for session {session_id}')
            
            if task_id:
                with self.task_lock:
                    if task_id in self.cancelled_tasks:
                        self.logger.log_task_status(task_id, 'cancelled', stage='Task was cancelled during initialization')
                        return None
                    
                    self.active_tasks[task_id]['status'] = 'processing'
                    self.active_tasks[task_id]['progress'] = 5
                    self.active_tasks[task_id]['stage'] = 'Initializing model and parameters'
                    self.active_tasks[task_id]['session_id'] = session_id
            
            # Convert and log parameters
            params_log = {
                'output_type': int(str(params.get('output_type', '0'))),
                'input_type': int(str(params.get('input_type', '0'))),
                'classification_threshold': float(str(params.get('classification_threshold', '0.35'))),
                'prediction_threshold': float(str(params.get('prediction_threshold', '0.5'))),
                'save_labeled_image': str(params.get('save_labeled_image', 'false')).lower() == 'true',
                'yolo_model_type': str(params.get('yolo_model_type', 'n'))
            }
            self.logger.log_system(f'Task {task_id} parameters: {params_log}')
            
            # Execute model
            output_folder = execute_func(
                input_folder,
                params_log['input_type'],
                params_log['classification_threshold'],
                params_log['prediction_threshold'],
                params_log['save_labeled_image'],
                params_log['output_type'],
                params_log['yolo_model_type']
            )
            
            if not output_folder or not os.path.exists(output_folder):
                raise Exception("Output folder not found or not returned by model execution")
            
            self.logger.log_file_operation('CREATE', output_folder)
            
            if task_id:
                with self.task_lock:
                    self.active_tasks[task_id]['progress'] = 95
                    self.active_tasks[task_id]['stage'] = 'Creating ZIP file'
            
            # Create ZIP file
            zip_name = f"{os.path.basename(output_folder)}.zip"
            zip_path = os.path.join(os.path.dirname(output_folder), zip_name)
            
            self.logger.log_system(f'Creating ZIP file for task {task_id}')
            
            if os.path.exists(zip_path):
                os.remove(zip_path)
                self.logger.log_cleanup('old_zip', zip_path)
            
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(output_folder):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, output_folder)
                        self.logger.log_file_operation('ZIP_ADD', arcname)
                        zipf.write(file_path, arcname)
            
            # Verify ZIP file
            if not os.path.exists(zip_path):
                raise Exception("ZIP file was not created")
            
            if os.path.getsize(zip_path) == 0:
                raise Exception("Created ZIP file is empty")
            
            with zipfile.ZipFile(zip_path, 'r') as zf:
                contents = zf.namelist()
                if not contents:
                    raise Exception("ZIP file has no contents")
                self.logger.log_file_operation('ZIP_VERIFY', zip_path, details=f"Files: {contents}")
                
                if zf.testzip() is not None:
                    raise Exception("ZIP file is corrupted")
            
            self.logger.log_file_operation('ZIP_CREATE', zip_path, details=f"Size: {os.path.getsize(zip_path)} bytes")
            
            # Remove original output folder
            shutil.rmtree(output_folder)
            self.logger.log_cleanup('output_folder', output_folder)
            
            # Update task status
            if task_id:
                with self.task_lock:
                    self.active_tasks[task_id]['status'] = 'completed'
                    self.active_tasks[task_id]['progress'] = 100
                    self.active_tasks[task_id]['stage'] = 'Completed'
                    self.active_tasks[task_id]['zip_path'] = zip_path
                    self.logger.log_task_status(task_id, 'completed', progress=100, stage='Completed')
            
            return task_id
            
        except Exception as e:
            self.logger.log_error(f'Error in process_task: {str(e)}', details=traceback.format_exc())
            if task_id:
                with self.task_lock:
                    self.active_tasks[task_id]['status'] = 'failed'
                    self.active_tasks[task_id]['error'] = str(e)
                    self.active_tasks[task_id]['stage'] = 'Failed'
                    self.logger.log_task_status(task_id, 'failed', error=str(e))
            
            if output_folder and os.path.exists(output_folder):
                shutil.rmtree(output_folder)
                self.logger.log_cleanup('failed_output', output_folder)
            
            raise
    
    def queue_processor(self, execute_func):
        """Process tasks from the queue."""
        self.logger.log_system('Queue processor started')
        while True:
            task = None
            try:
                task = self.task_queue.get()
                if task is None:  # Shutdown signal
                    self.logger.log_system('Queue processor received shutdown signal')
                    break
                
                self.logger.log_system(f'Processing queued task: {task["id"]}')
                self.process_task(task['id'], task['input_folder'], task['params'], execute_func)
                self.task_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.log_error(f'Task processing error: {str(e)}', details=traceback.format_exc())
                if task:
                    with self.task_lock:
                        task_id = task['id']
                        if task_id in self.active_tasks:
                            self.active_tasks[task_id]['status'] = 'failed'
                            self.active_tasks[task_id]['error'] = str(e)
                            self.active_tasks[task_id]['stage'] = 'Failed'
                            self.logger.log_task_status(task_id, 'failed', error=str(e))
                self.task_queue.task_done()
            finally:
                if task:
                    task_id = task['id']
                    def cleanup_task():
                        time.sleep(self.MAX_FILE_AGE_HOURS * 3600)
                        with self.task_lock:
                            if task_id in self.active_tasks:
                                del self.active_tasks[task_id]
                                self.logger.log_cleanup('task_data', f'Task {task_id}')
                    
                    cleanup_thread = threading.Thread(target=cleanup_task, daemon=True)
                    cleanup_thread.start()
    
    def cleanup_old_files(self):
        """Clean up files older than MAX_FILE_AGE_HOURS for input and MAX_OUTPUT_AGE_HOURS for output."""
        self.logger.log_system('File cleanup service started')
        while True:
            try:
                current_time = datetime.now()
                
                # Clean up input folders
                for folder in os.listdir(self.BASE_UPLOAD_FOLDER):
                    path = os.path.join(self.BASE_UPLOAD_FOLDER, folder)
                    if os.path.isdir(path):
                        try:
                            folder_time = datetime.strptime(folder.split('_')[0] + '_' + folder.split('_')[1], '%Y%m%d_%H%M%S')
                            if current_time - folder_time > timedelta(hours=self.MAX_FILE_AGE_HOURS):
                                shutil.rmtree(path)
                                self.logger.log_cleanup('old_input', path)
                        except (ValueError, IndexError):
                            continue
                
                # Clean up output folders and ZIP files
                for item in os.listdir(self.BASE_OUTPUT_FOLDER):
                    path = os.path.join(self.BASE_OUTPUT_FOLDER, item)
                    try:
                        timestamp = item.split('_')[0] + '_' + item.split('_')[1]
                        folder_time = datetime.strptime(timestamp, '%Y%m%d_%H%M%S')
                        
                        if current_time - folder_time > timedelta(hours=self.MAX_OUTPUT_AGE_HOURS):
                            if os.path.isdir(path):
                                shutil.rmtree(path)
                                self.logger.log_cleanup('old_output_folder', path)
                            elif item.endswith('.zip'):
                                os.remove(path)
                                self.logger.log_cleanup('old_zip', path)
                    except (ValueError, IndexError):
                        continue
            
            except Exception as e:
                self.logger.log_error(f'Cleanup error: {str(e)}', details=traceback.format_exc())
            
            time.sleep(1800)  # Run every 30 minutes 