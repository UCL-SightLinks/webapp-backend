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
        
        # Create base directories
        for folder in [self.BASE_UPLOAD_FOLDER, self.BASE_OUTPUT_FOLDER]:
            if not os.path.exists(folder):
                os.makedirs(folder)
    
    def add_task(self, task_data):
        """Add a new task to active tasks."""
        task_id = str(uuid.uuid4())
        with self.task_lock:
            self.active_tasks[task_id] = task_data
        return task_id
    
    def queue_task(self, task):
        """Add a task to the processing queue."""
        self.task_queue.put(task)
    
    def get_task_status(self, task_id):
        """Get the status of a task."""
        with self.task_lock:
            return self.active_tasks.get(task_id)
    
    def cancel_task(self, task_id):
        """Cancel a task if it's still running or queued.
        
        Args:
            task_id (str): The ID of the task to cancel
            
        Returns:
            bool: True if task was cancelled, False if task not found or already completed
        """
        with self.task_lock:
            task = self.active_tasks.get(task_id)
            if not task:
                return False
                
            # Can only cancel tasks that are queued or processing
            if task['status'] not in ['queued', 'processing']:
                return False
                
            # Mark task as cancelled
            self.cancelled_tasks.add(task_id)
            task['status'] = 'cancelled'
            task['stage'] = 'Cancelled by user'
            task['progress'] = 100
            
            # Clean up task resources
            input_folder = task.get('input_folder')
            if input_folder and os.path.exists(input_folder):
                try:
                    shutil.rmtree(input_folder)
                    print(f"Cleaned up input folder for cancelled task: {input_folder}")
                except Exception as e:
                    print(f"Error cleaning up input folder: {str(e)}")
            
            print(f"Task {task_id} cancelled successfully")
            return True
    
    def process_task(self, task_id, input_folder, params, execute_func):
        """Process a single task and update its status."""
        output_folder = None
        try:
            # Check if task was cancelled
            if task_id in self.cancelled_tasks:
                print(f"Task {task_id} was cancelled, skipping processing")
                return None
                
            print(f"\n=== Processing Task {task_id} ===")
            print("Input Folder:", input_folder)
            print("Raw Parameters:", params)
            
            # Get session_id from input folder path
            session_id = os.path.basename(input_folder)
            print("Session ID:", session_id)
            
            # Initialize task (5%)
            if task_id:
                with self.task_lock:
                    # Check again for cancellation
                    if task_id in self.cancelled_tasks:
                        print(f"Task {task_id} was cancelled during initialization")
                        return None
                        
                    self.active_tasks[task_id]['status'] = 'processing'
                    self.active_tasks[task_id]['progress'] = 5
                    self.active_tasks[task_id]['stage'] = 'Initializing model and parameters'
                    self.active_tasks[task_id]['session_id'] = session_id
            
            # Convert parameters and execute model
            output_type = int(str(params.get('output_type', '0')))
            input_type = int(str(params.get('input_type', '0')))
            classification_threshold = float(str(params.get('classification_threshold', '0.35')))
            prediction_threshold = float(str(params.get('prediction_threshold', '0.5')))
            save_labeled_image = str(params.get('save_labeled_image', 'false')).lower() == 'true'
            yolo_model_type = str(params.get('yolo_model_type', 'n'))
            
            # Execute model
            output_folder = execute_func(
                input_folder,
                input_type,
                classification_threshold,
                prediction_threshold,
                save_labeled_image,
                output_type,
                yolo_model_type
            )
            
            if not output_folder or not os.path.exists(output_folder):
                raise Exception("Output folder not found or not returned by model execution")
            
            # Create ZIP file (95%)
            if task_id:
                with self.task_lock:
                    self.active_tasks[task_id]['progress'] = 95
                    self.active_tasks[task_id]['stage'] = 'Creating ZIP file'
            
            # Create ZIP file for output
            zip_name = f"{os.path.basename(output_folder)}.zip"
            zip_path = os.path.join(os.path.dirname(output_folder), zip_name)
            
            print("\n=== Creating ZIP File ===")
            print(f"Output folder: {output_folder}")
            print(f"Contents: {os.listdir(output_folder)}")
            print(f"ZIP path: {zip_path}")
            
            # Create ZIP file
            if os.path.exists(zip_path):
                os.remove(zip_path)
                
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(output_folder):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, output_folder)
                        print(f"Adding to ZIP: {arcname}")
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
                print(f"ZIP contents: {contents}")
                
                # Test ZIP integrity
                if zf.testzip() is not None:
                    raise Exception("ZIP file is corrupted")
            
            print(f"ZIP file created successfully: {zip_path}")
            print(f"ZIP file size: {os.path.getsize(zip_path)} bytes")
            
            # Remove original output folder
            shutil.rmtree(output_folder)
            
            # Update task status (100%)
            if task_id:
                with self.task_lock:
                    self.active_tasks[task_id]['status'] = 'completed'
                    self.active_tasks[task_id]['progress'] = 100
                    self.active_tasks[task_id]['stage'] = 'Completed'
                    self.active_tasks[task_id]['zip_path'] = zip_path
            
            return task_id
            
        except Exception as e:
            print(f"Error in process_task: {str(e)}")
            if task_id:
                with self.task_lock:
                    self.active_tasks[task_id]['status'] = 'failed'
                    self.active_tasks[task_id]['error'] = str(e)
                    self.active_tasks[task_id]['stage'] = 'Failed'
            
            # Clean up
            if output_folder and os.path.exists(output_folder):
                shutil.rmtree(output_folder)
            
            raise
    
    def queue_processor(self, execute_func):
        """Process tasks from the queue."""
        while True:
            task = None
            try:
                task = self.task_queue.get()
                if task is None:  # Shutdown signal
                    break
                
                self.process_task(task['id'], task['input_folder'], task['params'], execute_func)
                self.task_queue.task_done()
            
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Task processing error: {str(e)}")
                if task:
                    with self.task_lock:
                        task_id = task['id']
                        if task_id in self.active_tasks:
                            self.active_tasks[task_id]['status'] = 'failed'
                            self.active_tasks[task_id]['error'] = str(e)
                            self.active_tasks[task_id]['stage'] = 'Failed'
                self.task_queue.task_done()
            finally:
                # Clean up task from active_tasks after MAX_FILE_AGE_HOURS
                if task:
                    task_id = task['id']
                    def cleanup_task():
                        time.sleep(self.MAX_FILE_AGE_HOURS * 3600)  # Convert hours to seconds
                        with self.task_lock:
                            if task_id in self.active_tasks:
                                del self.active_tasks[task_id]
                                print(f"Cleaned up task {task_id} from active tasks")
                    
                    cleanup_thread = threading.Thread(target=cleanup_task, daemon=True)
                    cleanup_thread.start()
    
    def cleanup_old_files(self):
        """Clean up files older than MAX_FILE_AGE_HOURS for input and MAX_OUTPUT_AGE_HOURS for output."""
        while True:
            try:
                current_time = datetime.now()
                
                # Clean up input folders (2 hours)
                for folder in os.listdir(self.BASE_UPLOAD_FOLDER):
                    path = os.path.join(self.BASE_UPLOAD_FOLDER, folder)
                    if os.path.isdir(path):
                        try:
                            folder_time = datetime.strptime(folder.split('_')[0] + '_' + folder.split('_')[1], '%Y%m%d_%H%M%S')
                            if current_time - folder_time > timedelta(hours=self.MAX_FILE_AGE_HOURS):
                                shutil.rmtree(path)
                                print(f"Cleaned up input folder: {path}")
                        except (ValueError, IndexError):
                            continue
                
                # Clean up output folders and ZIP files (4 hours)
                for item in os.listdir(self.BASE_OUTPUT_FOLDER):
                    path = os.path.join(self.BASE_OUTPUT_FOLDER, item)
                    try:
                        # Parse the timestamp from the name
                        timestamp = item.split('_')[0] + '_' + item.split('_')[1]
                        folder_time = datetime.strptime(timestamp, '%Y%m%d_%H%M%S')
                        
                        # Check if item is old enough to clean up (4 hours for both folders and zips)
                        if current_time - folder_time > timedelta(hours=self.MAX_OUTPUT_AGE_HOURS):
                            if os.path.isdir(path):
                                shutil.rmtree(path)
                                print(f"Cleaned up old output folder: {path}")
                            elif item.endswith('.zip'):
                                os.remove(path)
                                print(f"Cleaned up old ZIP file: {path}")
                    except (ValueError, IndexError):
                        continue
            
            except Exception as e:
                print(f"Cleanup error: {str(e)}")
            
            time.sleep(1800)  # Run every 30 minutes 