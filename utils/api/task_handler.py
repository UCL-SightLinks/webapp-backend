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
from concurrent.futures import ThreadPoolExecutor, Future
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
        self.MAX_CONCURRENT_TASKS = 5
        
        # Task queue and tracking
        self.task_queue = queue.Queue(maxsize=self.MAX_QUEUE_SIZE)
        self.active_tasks = {}
        self.task_lock = threading.Lock()
        self.cancelled_tasks = set()  # Track cancelled tasks
        self.task_events = {}  # Track cancellation events for each task
        
        # Thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=self.MAX_CONCURRENT_TASKS)
        self.processing_tasks = set()  # Track currently processing tasks
        
        # Initialize logger
        self.logger = LoggerHandler()
        
        # Create base directories
        for folder in [self.BASE_UPLOAD_FOLDER, self.BASE_OUTPUT_FOLDER]:
            if not os.path.exists(folder):
                os.makedirs(folder)
                self.logger.log_system(f'Created base directory: {folder}')
    
    def can_accept_task(self):
        """Check if we can accept a new task."""
        with self.task_lock:
            return len(self.processing_tasks) < self.MAX_CONCURRENT_TASKS
    
    def add_task(self, task_data):
        """Add a new task to active tasks."""
        task_id = str(uuid.uuid4())
        with self.task_lock:
            self.active_tasks[task_id] = task_data
            # Create cancellation event for the task
            self.task_events[task_id] = threading.Event()
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
            
            # Set cancellation flag and event
            self.cancelled_tasks.add(task_id)
            if task_id in self.task_events:
                self.task_events[task_id].set()
            
            task['status'] = 'cancelled'
            task['stage'] = 'Cancelled by user'
            task['progress'] = 100
            
            if task_id in self.processing_tasks:
                self.processing_tasks.remove(task_id)
                # If there's a future associated with this task, try to cancel it
                if 'future' in task:
                    future = task['future']
                    if future and not future.done():
                        future.cancel()
                        # Wait a short time for the task to clean up
                        try:
                            future.result(timeout=1.0)
                        except:
                            pass  # Ignore any exceptions from the cancelled task
                    del task['future']
            
            # Clean up input folder
            input_folder = task.get('input_folder')
            if input_folder and os.path.exists(input_folder):
                try:
                    shutil.rmtree(input_folder)
                    self.logger.log_cleanup('input_folder', input_folder)
                except Exception as e:
                    self.logger.log_error(f'Error cleaning up input folder: {str(e)}')
            
            self.logger.log_task_status(task_id, 'cancelled', progress=100, stage='Cancelled by user')
            return True
    
    def check_cancellation(self, task_id):
        """Check if a task has been cancelled."""
        if task_id in self.cancelled_tasks:
            return True
        if task_id in self.task_events and self.task_events[task_id].is_set():
            return True
        return False
    
    def process_task(self, task_id, input_folder, params, execute_func):
        """Process a single task and update its status."""
        output_folder = None
        try:
            if self.check_cancellation(task_id):
                self.logger.log_task_status(task_id, 'cancelled', stage='Task was cancelled before processing')
                # Clean up input folder on cancellation
                if input_folder and os.path.exists(input_folder):
                    shutil.rmtree(input_folder)
                    self.logger.log_cleanup('input_folder', input_folder)
                return None
            
            self.logger.log_task_status(task_id, 'processing', progress=0, stage='Starting task initialization')
            
            session_id = os.path.basename(input_folder)
            self.logger.log_system(f'Processing task {task_id} for session {session_id}')
            
            if task_id:
                with self.task_lock:
                    if self.check_cancellation(task_id):
                        self.logger.log_task_status(task_id, 'cancelled', stage='Task was cancelled during initialization')
                        # Clean up input folder on cancellation
                        if input_folder and os.path.exists(input_folder):
                            shutil.rmtree(input_folder)
                            self.logger.log_cleanup('input_folder', input_folder)
                        return None
                    
                    self.active_tasks[task_id]['status'] = 'processing'
                    self.active_tasks[task_id]['progress'] = 2
                    self.active_tasks[task_id]['stage'] = 'Initializing processing environment'
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
            
            # Execute model with progress callback and cancellation check
            def progress_callback(stage, progress):
                if task_id:
                    # Check for cancellation
                    if self.check_cancellation(task_id):
                        # Clean up input folder on cancellation
                        if input_folder and os.path.exists(input_folder):
                            shutil.rmtree(input_folder)
                            self.logger.log_cleanup('input_folder', input_folder)
                        raise Exception("Task cancelled by user")
                    
                    with self.task_lock:
                        self.active_tasks[task_id]['progress'] = progress
                        self.active_tasks[task_id]['stage'] = stage
                        self.logger.log_task_status(task_id, 'processing', progress=progress, stage=stage)
            
            # Check for cancellation before starting execution
            if self.check_cancellation(task_id):
                # Clean up input folder on cancellation
                if input_folder and os.path.exists(input_folder):
                    shutil.rmtree(input_folder)
                    self.logger.log_cleanup('input_folder', input_folder)
                raise Exception("Task cancelled by user")
            
            # Pass cancellation event to execute function
            output_folder = execute_func(
                input_folder,
                params_log['input_type'],
                params_log['classification_threshold'],
                params_log['prediction_threshold'],
                params_log['save_labeled_image'],
                params_log['output_type'],
                params_log['yolo_model_type'],
                progress_callback,
                lambda: self.check_cancellation(task_id)  # Pass cancellation check function
            )
            
            # Check for cancellation after execution
            if self.check_cancellation(task_id):
                # Clean up both input and output folders on cancellation
                if input_folder and os.path.exists(input_folder):
                    shutil.rmtree(input_folder)
                    self.logger.log_cleanup('input_folder', input_folder)
                if output_folder and os.path.exists(output_folder):
                    shutil.rmtree(output_folder)
                    self.logger.log_cleanup('output_folder', output_folder)
                raise Exception("Task cancelled by user")
            
            if not output_folder or not os.path.exists(output_folder):
                raise Exception("Output folder not found or not returned by model execution")
            
            self.logger.log_file_operation('CREATE', output_folder)
            
            if task_id:
                with self.task_lock:
                    if self.check_cancellation(task_id):
                        # Clean up both input and output folders on cancellation
                        if input_folder and os.path.exists(input_folder):
                            shutil.rmtree(input_folder)
                            self.logger.log_cleanup('input_folder', input_folder)
                        if output_folder and os.path.exists(output_folder):
                            shutil.rmtree(output_folder)
                            self.logger.log_cleanup('output_folder', output_folder)
                        raise Exception("Task cancelled by user")
                        
                    self.active_tasks[task_id]['progress'] = 95
                    self.active_tasks[task_id]['stage'] = 'Creating final ZIP archive'
                    self.logger.log_task_status(task_id, 'processing', progress=95, stage='Creating final ZIP archive')
            
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
                        if self.check_cancellation(task_id):
                            # Clean up all resources on cancellation
                            if input_folder and os.path.exists(input_folder):
                                shutil.rmtree(input_folder)
                                self.logger.log_cleanup('input_folder', input_folder)
                            if output_folder and os.path.exists(output_folder):
                                shutil.rmtree(output_folder)
                                self.logger.log_cleanup('output_folder', output_folder)
                            if os.path.exists(zip_path):
                                os.remove(zip_path)
                                self.logger.log_cleanup('zip_file', zip_path)
                            raise Exception("Task cancelled during ZIP creation")
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, output_folder)
                        self.logger.log_file_operation('ZIP_ADD', arcname)
                        zipf.write(file_path, arcname)
            
            if task_id:
                with self.task_lock:
                    if self.check_cancellation(task_id):
                        # Clean up all resources on cancellation
                        if input_folder and os.path.exists(input_folder):
                            shutil.rmtree(input_folder)
                            self.logger.log_cleanup('input_folder', input_folder)
                        if output_folder and os.path.exists(output_folder):
                            shutil.rmtree(output_folder)
                            self.logger.log_cleanup('output_folder', output_folder)
                        if os.path.exists(zip_path):
                            os.remove(zip_path)
                            self.logger.log_cleanup('zip_file', zip_path)
                        raise Exception("Task cancelled after ZIP creation")
                        
                    self.active_tasks[task_id]['status'] = 'completed'
                    self.active_tasks[task_id]['progress'] = 100
                    self.active_tasks[task_id]['stage'] = 'Task completed successfully'
                    self.active_tasks[task_id]['zip_path'] = zip_path
                    self.logger.log_task_status(task_id, 'completed', progress=100, stage='Task completed successfully')
            
            return task_id
            
        except Exception as e:
            self.logger.log_error(f'Task processing failed: {str(e)}')
            if task_id:
                with self.task_lock:
                    if self.check_cancellation(task_id):
                        self.active_tasks[task_id]['status'] = 'cancelled'
                        self.active_tasks[task_id]['stage'] = 'Task cancelled by user'
                        self.logger.log_task_status(task_id, 'cancelled', stage='Task cancelled by user')
                    else:
                        self.active_tasks[task_id]['status'] = 'failed'
                        self.active_tasks[task_id]['error'] = str(e)
                        self.active_tasks[task_id]['stage'] = f'Processing failed: {str(e)}'
                        self.logger.log_task_status(task_id, 'failed', error=str(e))
            
            # Clean up any remaining resources on error
            if input_folder and os.path.exists(input_folder):
                shutil.rmtree(input_folder)
                self.logger.log_cleanup('input_folder', input_folder)
            if output_folder and os.path.exists(output_folder):
                shutil.rmtree(output_folder)
                self.logger.log_cleanup('output_folder', output_folder)
            
            raise e
        finally:
            # Clean up cancellation event
            with self.task_lock:
                if task_id in self.task_events:
                    del self.task_events[task_id]
    
    def queue_processor(self, execute_func):
        """Process tasks from the queue using thread pool."""
        self.logger.log_system('Queue processor started')
        while True:
            try:
                task = self.task_queue.get()
                if task is None:  # Shutdown signal
                    self.logger.log_system('Queue processor received shutdown signal')
                    break
                
                # Wait until we can accept a new task
                while not self.can_accept_task():
                    time.sleep(1)
                
                # Submit task to thread pool
                with self.task_lock:
                    self.processing_tasks.add(task['id'])
                
                def task_wrapper(task):
                    try:
                        self.process_task(task['id'], task['input_folder'], task['params'], execute_func)
                    except Exception as e:
                        if task['id'] in self.cancelled_tasks:
                            self.logger.log_system(f'Task {task["id"]} was cancelled')
                        else:
                            self.logger.log_error(f'Task processing error: {str(e)}')
                    finally:
                        with self.task_lock:
                            if task['id'] in self.processing_tasks:
                                self.processing_tasks.remove(task['id'])
                        self.task_queue.task_done()
                
                future = self.thread_pool.submit(task_wrapper, task)
                
                # Store the future in the task data for cancellation
                with self.task_lock:
                    self.active_tasks[task['id']]['future'] = future
            
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.log_error(f'Queue processor error: {str(e)}', details=traceback.format_exc())
                if task:
                    with self.task_lock:
                        if task['id'] in self.processing_tasks:
                            self.processing_tasks.remove(task['id'])
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