"""Task handler module for managing task processing and cleanup."""

import os
import sys
import time
import json
import uuid
import shutil
import threading
import zipfile
import queue
import psutil
import rasterio
import traceback
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, Future
from utils.compress import compress_folder_to_zip
from utils.api.logger_handler import LoggerHandler

class TaskHandler:
    """Handles task processing, queuing, and cleanup."""

    def __init__(self):
        """Initialize task handler with configuration."""
        self.BASE_UPLOAD_FOLDER = 'input'
        self.BASE_OUTPUT_FOLDER = 'run/output'
        self.BASE_EXTRACT_FOLDER = 'run/extract'
        self.MAX_FILE_AGE_HOURS = 2  # For input files
        self.MAX_OUTPUT_AGE_HOURS = 2  # For output files and zips
        self.MAX_QUEUE_SIZE = 10
        self.MAX_CONCURRENT_TASKS = 5
        
        # Setup logger
        self.logger = LoggerHandler()
        
        # Tasks persistence file - use absolute path
        self.TASKS_DB_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'tasks_db.json')
        self.logger.log_system(f"Using tasks database file: {self.TASKS_DB_FILE}")

        # Task queue and tracking
        self.task_queue = queue.Queue(maxsize=self.MAX_QUEUE_SIZE)
        self.active_tasks = {}
        self.task_lock = threading.Lock()
        self.cancelled_tasks = set()  # Track cancelled tasks
        self.task_events = {}  # Track cancellation events for each task
        
        # Load existing tasks from disk if available
        self._load_tasks()

        # Server statistics
        self.stats = {
            'total_tasks_processed': 0,
            'total_files_processed': 0,
            'failed_tasks': 0,
            'cancelled_tasks': 0,
            'start_time': datetime.now(),
            'current_tasks': 0,
            'queued_tasks': 0
        }
        self.stats_lock = threading.Lock()

        # Thread control
        self.shutdown_flag = threading.Event()
        self.is_running = False

        # CPU usage tracking
        self.process = psutil.Process(os.getpid())
        self.cpu_count = psutil.cpu_count() or 1
        self.last_cpu_check = None
        self.last_cpu_percent = 0

        # Thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=self.MAX_CONCURRENT_TASKS)
        self.processing_tasks = set()  # Track currently processing tasks

        # Create base directories
        for folder in [self.BASE_UPLOAD_FOLDER, self.BASE_OUTPUT_FOLDER]:
            if not os.path.exists(folder):
                os.makedirs(folder)
                self.logger.log_system(f'Created base directory: {folder}')

    def update_stats(self, stat_key, value=1, increment=True):
        """Update server statistics."""
        with self.stats_lock:
            if increment:
                self.stats[stat_key] += value
            else:
                self.stats[stat_key] = value

    def get_server_status(self):
        """Get current server status and statistics."""
        with self.stats_lock:
            with self.task_lock:
                # Count tasks by their actual status
                queued_tasks = [task for task in self.active_tasks.values()
                              if task.get('status') == 'queued']
                processing_tasks = [task for task in self.active_tasks.values()
                                  if task.get('status') == 'processing']

                queued_count = len(queued_tasks)
                processing_count = len(processing_tasks)

                # Update stats with accurate counts
                self.stats['queued_tasks'] = queued_count
                self.stats['current_tasks'] = processing_count

                # Create status response
                status = self.stats.copy()
                status.update({
                    'uptime_seconds': (datetime.now() - status['start_time']).total_seconds(),
                    'max_concurrent_tasks': self.MAX_CONCURRENT_TASKS,
                    'max_queue_size': self.MAX_QUEUE_SIZE,
                    'memory_usage_mb': self._get_memory_usage(),
                    'cpu_usage_percent': self._get_cpu_usage(),
                    'active_tasks': len(self.active_tasks),
                    'processing_tasks': processing_count,
                    'queue_size': queued_count,
                    'queued_task_ids': [task.get('id') for task in queued_tasks],
                    'processing_task_ids': [task.get('id') for task in processing_tasks]
                })
                return status

    def _get_memory_usage(self):
        """Get current memory usage in MB."""
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0

    def _get_cpu_usage(self):
        """Get current CPU usage percentage accurately (0-100%)."""
        try:
            current_time = time.time()

            # Only update every second to avoid excessive CPU usage from checking
            if self.last_cpu_check is None or (current_time - self.last_cpu_check) >= 1.0:
                # Get CPU percent for this process
                process_percent = self.process.cpu_percent() / self.cpu_count

                # Get system-wide CPU percent
                system_percent = psutil.cpu_percent(interval=None)

                # Use the higher of the two values, but ensure it's between 0-100
                self.last_cpu_percent = min(100, max(0, max(process_percent, system_percent)))
                self.last_cpu_check = current_time

            return self.last_cpu_percent
        except Exception as e:
            self.logger.log_error(f"Error getting CPU usage: {str(e)}")
            return 0

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
        with self.task_lock:
            task_id = task['id']
            # Get current queue size before adding new task
            current_queue_size = len([t for t in self.active_tasks.values()
                                    if t.get('status') == 'queued'])

            # Initialize task in active_tasks before queuing
            self.active_tasks[task_id] = {
                'id': task_id,
                'status': 'queued',
                'progress': 0,
                'stage': 'Waiting in queue',
                'created_at': datetime.now(),
                'session_id': os.path.basename(task['input_folder']),
                'input_folder': task['input_folder'],
                'is_cancelled': False,
                'queue_position': current_queue_size + 1
            }

            # Create cancellation event if needed
            if task_id not in self.task_events:
                self.task_events[task_id] = threading.Event()

            # Update queue stats based on actual task status
            with self.stats_lock:
                # Count actual queued tasks
                self.stats['queued_tasks'] = current_queue_size + 1
                self.logger.log_system(f'Task {task_id} queued (position {current_queue_size + 1}, total queued: {current_queue_size + 1})')

            # Put task in queue AFTER updating status
            self.task_queue.put(task)
            self.logger.log_task_status(task_id, 'queued', stage=f'Added to queue (position {current_queue_size + 1})')

            # Update queue positions for all queued tasks
            self._update_queue_positions()

    def get_task_status(self, task_id):
        """Get the status of a task."""
        if not task_id:
            return {
                'status': 'unknown',
                'error': 'Invalid task ID'
            }

        with self.task_lock:
            # First check if task is in memory
            task = self.active_tasks.get(task_id)
            
            # If not in memory, try to load it from the persistent file
            if not task:
                try:
                    if os.path.exists(self.TASKS_DB_FILE):
                        with open(self.TASKS_DB_FILE, 'r') as f:
                            tasks_db = json.load(f)
                            task = tasks_db.get(task_id)
                            
                            if task:
                                self.logger.log_system(f"Retrieved task {task_id} from disk")
                                # Convert ISO datetime strings back to datetime objects
                                if 'created_at' in task and isinstance(task['created_at'], str):
                                    try:
                                        task['created_at'] = datetime.fromisoformat(task['created_at'])
                                    except ValueError:
                                        task['created_at'] = datetime.now()
                                
                                # Add back to active_tasks if it's a completed task with outputs
                                if task.get('status') == 'completed' and (task.get('output_folder') or task.get('zip_path')):
                                    self.active_tasks[task_id] = task
                except Exception as e:
                    self.logger.log_error(f"Error loading task {task_id} from file: {str(e)}")
            
            if not task:
                return {
                    'status': 'unknown',
                    'error': 'Task not found'
                }
            
            # Ensure session_id is set
            if 'session_id' not in task:
                # Try to get session_id from input_folder
                if 'input_folder' in task:
                    task['session_id'] = os.path.basename(task['input_folder'])
                else:
                    # Generate a new session_id if none exists
                    task['session_id'] = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
            
            return task.copy()  # Return a copy to prevent external modification

    def _get_queue_position(self, task_id):
        """Get the position of a task in the queue (1-based)."""
        queued_tasks = [t for t in self.active_tasks.values() if t.get('status') == 'queued']
        for i, task in enumerate(queued_tasks, 1):
            if task.get('id') == task_id:
                return i
        return 0

    def cancel_task(self, task_id):
        """Cancel a task if it's still running or queued."""
        with self.task_lock:
            task = self.active_tasks.get(task_id)
            if not task:
                # Check if task is in processing_tasks set
                if task_id in self.processing_tasks:
                    # Task exists but state was lost, recreate it
                    task = {
                        'id': task_id,
                        'status': 'processing',
                        'progress': 0,
                        'stage': 'Processing',
                        'created_at': datetime.now(),
                        'is_cancelled': False
                    }
                    self.active_tasks[task_id] = task
                else:
                    self.logger.log_error(f'Task not found for cancellation: {task_id}')
                    return False

            current_status = task.get('status', 'unknown')
            if current_status in ['completed', 'failed', 'cancelled']:
                self.logger.log_error(f'Cannot cancel task {task_id} in status: {current_status}')
                return False

            # Update stats
            with self.stats_lock:
                self.stats['cancelled_tasks'] += 1
                if current_status == 'queued':
                    self.stats['queued_tasks'] = max(0, self.stats['queued_tasks'] - 1)
                elif current_status == 'processing':
                    self.stats['current_tasks'] = max(0, self.stats['current_tasks'] - 1)

            # Mark task as cancelled
            task['is_cancelled'] = True
            task['status'] = 'cancelled'
            task['stage'] = 'Cancelled by user'
            task['progress'] = 100
            task['error'] = 'Task cancelled by user'

            # Set cancellation flag and event
            self.cancelled_tasks.add(task_id)
            if task_id in self.task_events:
                self.task_events[task_id].set()
            else:
                # Create event if it doesn't exist
                self.task_events[task_id] = threading.Event()
                self.task_events[task_id].set()

            # Handle running task
            if task_id in self.processing_tasks:
                self.processing_tasks.remove(task_id)
                if 'future' in task:
                    future = task['future']
                    if future and not future.done():
                        try:
                            future.cancel()
                            # Add a timeout to prevent hanging
                            future.result(timeout=1.0)
                        except Exception:
                            # Ignore any exceptions from cancellation
                            pass
                    if 'future' in task:
                        del task['future']

            # Clean up input folder if exists
            input_folder = task.get('input_folder')
            if input_folder and os.path.exists(input_folder):
                try:
                    shutil.rmtree(input_folder)
                    self.logger.log_cleanup('input_folder', input_folder)
                except Exception as e:
                    self.logger.log_error(f'Error cleaning up input folder: {str(e)}')

            # Clean up extract directory
            try:
                # Extract directory is named with the same timestamp as the task
                session_id = task.get('session_id')
                if session_id:
                    # Extract timestamp from session_id (format: YYYYMMDD_HHMMSS_uuid)
                    timestamp = '_'.join(session_id.split('_')[:2])
                    extract_base_dir = 'run/extract'
                    
                    # Look for extract directories with matching timestamp
                    if os.path.exists(extract_base_dir):
                        for dir_name in os.listdir(extract_base_dir):
                            if dir_name.startswith(timestamp):
                                extract_dir = os.path.join(extract_base_dir, dir_name)
                                if os.path.exists(extract_dir):
                                    shutil.rmtree(extract_dir)
                                    self.logger.log_cleanup('extract_dir', extract_dir)
            except Exception as e:
                self.logger.log_error(f'Error cleaning up extract directory: {str(e)}')

            self.logger.log_task_status(task_id, 'cancelled', progress=100, stage='Cancelled by user')
            return True

    def check_cancellation(self, task_id):
        """Check if a task has been cancelled."""
        with self.task_lock:
            # First check if the task is marked as cancelled in active_tasks
            if task_id in self.active_tasks:
                return self.active_tasks[task_id].get('is_cancelled', False)
            # Then check if it's in the cancelled_tasks set
            return task_id in self.cancelled_tasks

    def process_task(self, task_id, input_folder, params, execute_func):
        """Process a task with the given parameters."""
        logger_handler = self.logger
        
        try:
            # Get session_id from input_folder
            session_id = os.path.basename(input_folder)

            # Log task start
            logger_handler.log_system("\n=== Starting Task Processing ===")
            logger_handler.log_system(f"Task ID: {task_id}")
            logger_handler.log_system(f"Session ID: {session_id}")
            logger_handler.log_system(f"Input folder: {input_folder}")
            logger_handler.log_system(f"Parameters: {params}")
            
            # Update task status with session_id
            with self.task_lock:
                if task_id in self.active_tasks:
                    self.active_tasks[task_id].update({
                        'status': 'processing',
                        'session_id': session_id,
                        'input_folder': input_folder
                    })
            
            # Log input folder contents and validate TIF files
            logger_handler.log_system("\n=== Input Folder Contents ===")
            for item in os.listdir(input_folder):
                item_path = os.path.join(input_folder, item)
                if os.path.isfile(item_path):
                    size = os.path.getsize(item_path)
                    logger_handler.log_system(f"\nFile: {item}")
                    logger_handler.log_system(f"Full path: {item_path}")
                    logger_handler.log_system(f"Size: {size} bytes")
                    
                    # Validate TIF files
                    if item.lower().endswith(('.tif', '.tiff')):
                        try:
                            # First check if file exists and has content
                            if not os.path.exists(item_path):
                                raise ValueError(f"TIF file does not exist: {item_path}")
                                
                            if size == 0:
                                raise ValueError(f"TIF file has 0 bytes: {item_path}")
                            
                            # Try to read the file with rasterio
                            import rasterio
                            with rasterio.open(item_path) as src:
                                logger_handler.log_system("\nTIF file validation:")
                                logger_handler.log_system(f"- Width: {src.width}")
                                logger_handler.log_system(f"- Height: {src.height}")
                                logger_handler.log_system(f"- Bands: {src.count}")
                                logger_handler.log_system(f"- CRS: {src.crs}")
                                logger_handler.log_system(f"- Transform: {src.transform}")
                                
                                # Additional validation
                                if src.count not in [1, 3, 4]:
                                    raise ValueError(f"Invalid number of bands in TIF file: {src.count}")
                                    
                                if src.width == 0 or src.height == 0:
                                    raise ValueError("TIF file has zero dimensions")
                                    
                        except Exception as e:
                            logger_handler.log_error(f"Error validating TIF file {item}: {str(e)}")
                            raise ValueError(f"Invalid TIF file {item}: {str(e)}")
                else:
                    logger_handler.log_system(f"\nDirectory: {item}")
            
            logger_handler.log_system("\n=== Executing Task ===")
            
            # Execute the task with the input folder containing the saved files
            try:
                # Convert parameters to their proper types
                input_type = str(params['input_type'])
                classification_threshold = float(params['classification_threshold'])
                prediction_threshold = float(params['prediction_threshold'])
                save_labeled_image = params['save_labeled_image'].lower() == 'true' if isinstance(params['save_labeled_image'], str) else bool(params['save_labeled_image'])
                output_type = str(params['output_type'])
                yolo_model_type = str(params['yolo_model_type'])
                
                logger_handler.log_system("\nExecuting main processing function with parameters:")
                logger_handler.log_system(f"- Input folder: {input_folder}")
                logger_handler.log_system(f"- Input type: {input_type}")
                logger_handler.log_system(f"- Classification threshold: {classification_threshold}")
                logger_handler.log_system(f"- Prediction threshold: {prediction_threshold}")
                logger_handler.log_system(f"- Save labeled image: {save_labeled_image}")
                logger_handler.log_system(f"- Output type: {output_type}")
                logger_handler.log_system(f"- YOLO model type: {yolo_model_type}")
                
                # Check if yolo model file exists
                models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))
                model_path = os.path.join(models_dir, f"yolo-{yolo_model_type}.pt")
                logger_handler.log_system(f"Model path to check: {model_path}")
                logger_handler.log_system(f"Model exists: {os.path.exists(model_path)}")
                if os.path.exists(models_dir):
                    logger_handler.log_system(f"Available models: {os.listdir(models_dir)}")
                
                # Execute the main processing function
                output_folder = execute_func(
                    input_folder,
                    input_type,
                    classification_threshold,
                    prediction_threshold,
                    save_labeled_image,
                    output_type,
                    yolo_model_type
                )
                
                logger_handler.log_system(f"\nMain processing completed. Output folder: {output_folder}")
                
                # Update task status with output information
                with self.task_lock:
                    if task_id in self.active_tasks:
                        self.active_tasks[task_id].update({
                            'status': 'completed',
                            'progress': 100,
                            'stage': 'Completed',
                            'output_folder': output_folder,
                            'session_id': session_id  # Ensure session_id is preserved
                        })
                
                # Log task completion
                logger_handler.log_system("\n=== Task Completed Successfully ===")
                logger_handler.log_task_status(task_id, 'completed', stage='Task completed successfully')

                return task_id

            except Exception as e:
                logger_handler.log_error(f"Error executing task: {str(e)}")
                logger_handler.log_error(traceback.format_exc())
                raise

        except Exception as e:
            # Update task status with error information
            with self.task_lock:
                if task_id in self.active_tasks:
                    self.active_tasks[task_id].update({
                        'status': 'failed',
                        'error': str(e),
                        'stage': 'Failed',
                        'session_id': session_id  # Ensure session_id is preserved
                    })
            
            logger_handler.log_error(f"\n=== Task Failed ===")
            logger_handler.log_error(f"Error processing task {task_id}: {str(e)}", details=traceback.format_exc())
            raise

    def _execute_task(self, task_id, input_folder, params, execute_func):
        """Execute the task with proper progress tracking and cancellation checks."""
        try:
            # Check for cancellation before starting execution
            if self.check_cancellation(task_id):
                raise Exception("Task cancelled by user")

            # Extract timestamp from session_id for extract directory cleanup if needed
            session_id = os.path.basename(input_folder)
            extract_dir = None
            
            # Create a local wrapper to ensure we only pass the expected parameters
            # and properly handle cancellation during execution
            def safe_execute_wrapper(uploadDir, inputType, classificationThreshold, predictionThreshold, saveLabeledImage, outputType, yoloModelType):
                # Store the extract directory for cleanup if task is cancelled
                nonlocal extract_dir
                
                # Create a cancellation check function that will be called periodically
                def check_cancel():
                    # Check if task was cancelled during execution
                    if self.check_cancellation(task_id):
                        # If cancelled, clean up the extract directory
                        if extract_dir and os.path.exists(extract_dir):
                            try:
                                self.logger.log_system(f"Cleaning up extract directory due to cancellation: {extract_dir}")
                                shutil.rmtree(extract_dir)
                                self.logger.log_cleanup('extract_dir_during_execution', extract_dir)
                            except Exception as e:
                                self.logger.log_error(f"Error cleaning up extract directory during cancellation: {str(e)}")
                        
                        # Raise exception to terminate execution
                        raise Exception("Task cancelled by user during execution")
                
                # Create a wrapper around the execute function that periodically checks for cancellation
                result = None
                try:
                    # Patch the time.sleep function to check for cancellation periodically
                    original_sleep = time.sleep
                    
                    def patched_sleep(seconds):
                        # Break long sleeps into smaller chunks to check for cancellation more frequently
                        chunk_size = 0.5  # Check every half second
                        remaining = seconds
                        
                        while remaining > 0:
                            # Check for cancellation during sleep
                            try:
                                check_cancel()
                            except Exception as e:
                                # Restore original sleep before re-raising
                                time.sleep = original_sleep
                                raise e
                                
                            sleep_time = min(chunk_size, remaining)
                            original_sleep(sleep_time)
                            remaining -= sleep_time
                    
                    # Temporarily replace time.sleep with our patched version
                    time.sleep = patched_sleep
                    
                    # Set up the cancellation callback for the main execute function
                    import sys
                    sys._task_cancelled_callback = check_cancel
                    
                    # Check for cancellation before starting execution
                    check_cancel()
                    
                    # Execute the function, tracking any extract directory it creates
                    self.logger.log_system(f"\nExecuting main processing function with parameters:")
                    self.logger.log_system(f"- Upload directory: {uploadDir}")
                    self.logger.log_system(f"- Input type: {inputType}")
                    self.logger.log_system(f"- Classification threshold: {classificationThreshold}")
                    self.logger.log_system(f"- Prediction threshold: {predictionThreshold}")
                    self.logger.log_system(f"- Save labeled image: {saveLabeledImage}")
                    self.logger.log_system(f"- Output type: {outputType}")
                    self.logger.log_system(f"- YOLO model type: {yoloModelType}")
                    
                    result = execute_func(uploadDir, inputType, classificationThreshold, predictionThreshold, saveLabeledImage, outputType, yoloModelType)
                    
                    # Final cancellation check after execution
                    check_cancel()
                    
                    # Look for extract directory that might have been created
                    if not extract_dir:
                        timestamp = '_'.join(session_id.split('_')[:2])
                        extract_base_dir = 'run/extract'
                        
                        if os.path.exists(extract_base_dir):
                            for dir_name in os.listdir(extract_base_dir):
                                if dir_name.startswith(timestamp):
                                    extract_dir = os.path.join(extract_base_dir, dir_name)
                                    break
                    
                    return result
                    
                finally:
                    # Restore original sleep function
                    time.sleep = original_sleep
                    
                    # Remove the cancellation callback
                    if hasattr(sys, '_task_cancelled_callback'):
                        delattr(sys, '_task_cancelled_callback')
                    
                    # In case of any other exception, do a final cancellation check
                    try:
                        if self.check_cancellation(task_id):
                            raise Exception("Task cancelled by user during execution")
                    except Exception as e:
                        if "Task cancelled by user" in str(e):
                            raise e
                    
                    return result
            
            # Execute the task using the safe wrapper
            output_folder = safe_execute_wrapper(
                input_folder,
                params['input_type'],
                params['classification_threshold'],
                params['prediction_threshold'],
                params['save_labeled_image'],
                params['output_type'],
                params['yolo_model_type']
            )

            return output_folder

        except Exception as e:
            self.logger.log_error(f"Task execution failed: {str(e)}")
            
            # Clean up any possible lingering callback
            import sys
            if hasattr(sys, '_task_cancelled_callback'):
                delattr(sys, '_task_cancelled_callback')
            
            # If the task was cancelled, make sure the extract directory is cleaned up
            if self.check_cancellation(task_id):
                # Clean up extract directory if task was cancelled
                session_id = os.path.basename(input_folder)
                timestamp = '_'.join(session_id.split('_')[:2]) if session_id else ""
                
                if timestamp:
                    extract_base_dir = 'run/extract'
                    
                    if os.path.exists(extract_base_dir):
                        for dir_name in os.listdir(extract_base_dir):
                            if dir_name.startswith(timestamp):
                                extract_dir = os.path.join(extract_base_dir, dir_name)
                                if os.path.exists(extract_dir):
                                    try:
                                        self.logger.log_system(f"Cleaning up extract directory after cancellation: {extract_dir}")
                                        shutil.rmtree(extract_dir)
                                        self.logger.log_cleanup('extract_dir_after_cancel', extract_dir)
                                    except Exception as cleanup_err:
                                        self.logger.log_error(f"Error cleaning up extract directory after cancellation: {str(cleanup_err)}")

            raise

    def _create_zip_file(self, task_id, output_folder):
        """Create a ZIP file for the task."""
        try:
            # Get descriptor for the zip file name based on detection status
            descriptor = "detections"
            has_detections = True
            
            # Check if there were any detections by looking for the marker file
            no_detections_marker = os.path.join(output_folder, "no_detections.txt")
            if os.path.exists(no_detections_marker):
                descriptor = "no-detections"
                has_detections = False
            
            # Use a simple name format for the zip file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            zip_name = f"result_{timestamp}.zip"
            zip_path = os.path.join(os.path.dirname(output_folder), zip_name)

            self.logger.log_system(f'Creating ZIP file for task {task_id} at {zip_path}')

            if os.path.exists(zip_path):
                os.remove(zip_path)
                self.logger.log_cleanup('old_zip', zip_path)

            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Get all files in the output folder
                for root, _, files in os.walk(output_folder):
                    for file in files:
                        if self.check_cancellation(task_id):
                            # Clean up all resources on cancellation
                            if output_folder and os.path.exists(output_folder):
                                shutil.rmtree(output_folder)
                                self.logger.log_cleanup('output_folder', output_folder)
                            if os.path.exists(zip_path):
                                os.remove(zip_path)
                                self.logger.log_cleanup('zip_file', zip_path)
                            raise Exception("Task cancelled during ZIP creation")
                            
                        # Get the full file path
                        file_path = os.path.join(root, file)
                        
                        # Create a relative path for the archive
                        # This ensures files are at the right level without excessive nesting
                        rel_path = os.path.relpath(file_path, output_folder)
                        
                        # Log and add the file to the ZIP
                        self.logger.log_file_operation('ZIP_ADD', rel_path)
                        zipf.write(file_path, rel_path)
            
            # Verify the zip file was created properly
            if not os.path.exists(zip_path):
                raise Exception(f"ZIP file was not created at {zip_path}")
                
            # Verify zip file integrity
            with zipfile.ZipFile(zip_path, 'r') as verify_zip:
                # Test if any files in the archive are corrupted
                bad_file = verify_zip.testzip()
                if bad_file:
                    raise Exception(f"ZIP file is corrupted. Bad file: {bad_file}")
                
                # Verify the zip has contents
                if len(verify_zip.namelist()) == 0:
                    raise Exception("ZIP file is empty")
                    
            self.logger.log_system(f'Successfully created and verified ZIP file: {zip_path}')
            return zip_path

        except Exception as e:
            self.logger.log_error(f'Error creating ZIP file: {str(e)}')
            raise

    def queue_processor(self, execute_func):
        """Process tasks from the queue using thread pool."""
        self.logger.log_system('Queue processor started')
        self.is_running = True

        while not self.shutdown_flag.is_set():
            task = None
            try:
                # Use timeout to check shutdown flag periodically
                try:
                    task = self.task_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                if task is None:  # Shutdown signal
                    self.logger.log_system('Queue processor received shutdown signal')
                    break

                task_id = task['id']
                session_id = os.path.basename(task['input_folder'])

                # Wait until we can accept a new task
                while not self.can_accept_task() and not self.shutdown_flag.is_set():
                    time.sleep(1)
                    # Check if task was cancelled while waiting
                    if self.check_cancellation(task_id):
                        with self.task_lock:
                            if task_id in self.active_tasks:
                                self.active_tasks[task_id].update({
                                    'status': 'cancelled',
                                    'stage': 'Cancelled while waiting',
                                    'error': 'Task cancelled by user',
                                    'progress': 100,
                                    'session_id': session_id  # Ensure session_id is preserved
                                })
                                # Update queue stats when task is cancelled
                                with self.stats_lock:
                                    self.stats['queued_tasks'] = len([t for t in self.active_tasks.values()
                                                                    if t.get('status') == 'queued'])
                                    self.stats['current_tasks'] = len([t for t in self.active_tasks.values()
                                                                     if t.get('status') == 'processing'])
                                self._update_queue_positions()
                            self.task_queue.task_done()
                            continue

                if self.shutdown_flag.is_set():
                    break

                # Update task status to processing
                with self.task_lock:
                    if task_id in self.active_tasks:
                        self.processing_tasks.add(task_id)
                        self.active_tasks[task_id].update({
                            'status': 'processing',
                            'progress': 0,
                            'stage': 'Starting processing',
                            'session_id': session_id  # Ensure session_id is preserved
                        })
                        # Update queue stats when task starts processing
                        with self.stats_lock:
                            self.stats['queued_tasks'] = len([t for t in self.active_tasks.values()
                                                                    if t.get('status') == 'queued'])
                            self.stats['current_tasks'] = len([t for t in self.active_tasks.values()
                                                                     if t.get('status') == 'processing'])
                        # Update queue positions after removing task from queue
                        self._update_queue_positions()
                    else:
                        # Recreate task data if it was cleaned up
                        self.active_tasks[task_id] = {
                            'id': task_id,
                            'status': 'processing',
                            'progress': 0,
                            'stage': 'Starting processing',
                            'created_at': datetime.now(),
                            'session_id': session_id,
                            'input_folder': task['input_folder'],
                            'is_cancelled': False
                        }
                        self.processing_tasks.add(task_id)

                def task_wrapper(task):
                    try:
                        self.process_task(task['id'], task['input_folder'], task['params'], execute_func)
                    except Exception as e:
                        with self.task_lock:
                            if task['id'] in self.active_tasks:
                                if self.check_cancellation(task['id']):
                                    self.active_tasks[task['id']].update({
                                        'status': 'cancelled',
                                        'stage': 'Task was cancelled',
                                        'error': 'Task cancelled by user',
                                        'progress': 100
                                    })
                                else:
                                    self.active_tasks[task['id']].update({
                                        'status': 'failed',
                                        'stage': 'Task failed',
                                        'error': str(e),
                                        'progress': 100
                                    })
                    finally:
                        with self.task_lock:
                            if task['id'] in self.processing_tasks:
                                self.processing_tasks.remove(task['id'])
                        self.task_queue.task_done()

                future = self.thread_pool.submit(task_wrapper, task)

                # Store the future in the task data for cancellation
                with self.task_lock:
                    if task_id in self.active_tasks:
                        self.active_tasks[task_id]['future'] = future

            except Exception as e:
                self.logger.log_error(f'Queue processor error: {str(e)}', details=traceback.format_exc())
                if task:
                    task_id = task['id']
                    with self.task_lock:
                        if task_id in self.processing_tasks:
                            self.processing_tasks.remove(task_id)
                        if task_id in self.active_tasks:
                            self.active_tasks[task_id].update({
                                'status': 'failed',
                                'stage': 'Queue processing error',
                                'error': str(e),
                                'progress': 100
                            })
                    self.task_queue.task_done()

        self.is_running = False
        self.logger.log_system('Queue processor stopped')

    def cleanup_old_files(self):
        """Clean up files older than MAX_FILE_AGE_HOURS for input and MAX_OUTPUT_AGE_HOURS for output."""
        self.logger.log_system('File cleanup service started')

        while not self.shutdown_flag.is_set():
            try:
                current_time = datetime.now()
                self.logger.log_system(f'Running scheduled cleanup at {current_time.strftime("%Y-%m-%d %H:%M:%S")}')

                with self.task_lock:
                    # Get list of active tasks to prevent cleaning up in-use folders
                    active_input_folders = {
                        task.get('input_folder') for task in self.active_tasks.values()
                        if task.get('status') in ['queued', 'processing']
                    }
                    active_output_folders = {
                        os.path.dirname(task.get('zip_path', '')) for task in self.active_tasks.values()
                        if task.get('status') == 'completed' and task.get('zip_path')
                    }
                    active_extract_folders = {
                        task.get('extract_dir') for task in self.active_tasks.values()
                        if task.get('status') in ['queued', 'processing'] and task.get('extract_dir')
                    }

                    # Clean up old tasks from active_tasks ONLY if they are in a final state
                    current_tasks = list(self.active_tasks.items())
                    for task_id, task in current_tasks:
                        # Only clean up tasks that are in a final state and older than the threshold
                        if task.get('status') in ['completed', 'failed', 'cancelled']:
                            created_at = task.get('created_at')
                            if created_at and (current_time - created_at) > timedelta(hours=self.MAX_OUTPUT_AGE_HOURS):
                                # Double check task is not processing or queued
                                if task_id not in self.processing_tasks and task.get('status') not in ['queued', 'processing']:
                                    # Save all tasks before removing this one to ensure it's persisted
                                    self._save_tasks()
                                    
                                    del self.active_tasks[task_id]
                                    if task_id in self.cancelled_tasks:
                                        self.cancelled_tasks.remove(task_id)
                                    if task_id in self.task_events:
                                        del self.task_events[task_id]
                                    self.logger.log_cleanup('task_data', f'Task {task_id}')

                # Clean up input folders
                if os.path.exists(self.BASE_UPLOAD_FOLDER):
                    self.logger.log_system(f'Checking for old input folders in {self.BASE_UPLOAD_FOLDER}')
                    input_folders_cleaned = 0
                    for folder in os.listdir(self.BASE_UPLOAD_FOLDER):
                        if self.shutdown_flag.is_set():
                            break

                        path = os.path.join(self.BASE_UPLOAD_FOLDER, folder)
                        if path in active_input_folders:
                            continue

                        if os.path.isdir(path):
                            try:
                                # Check folder age by timestamp format or modification time
                                is_old = False
                                
                                # First try to parse from folder name
                                try:
                                    folder_time = datetime.strptime(folder.split('_')[0] + '_' + folder.split('_')[1], '%Y%m%d_%H%M%S')
                                    if current_time - folder_time > timedelta(hours=self.MAX_FILE_AGE_HOURS):
                                        is_old = True
                                except (ValueError, IndexError):
                                    # If we can't parse from name, use modification time
                                    mod_time = datetime.fromtimestamp(os.path.getmtime(path))
                                    if current_time - mod_time > timedelta(hours=self.MAX_FILE_AGE_HOURS):
                                        is_old = True
                                
                                if is_old:
                                    shutil.rmtree(path)
                                    input_folders_cleaned += 1
                                    self.logger.log_cleanup('old_input', path)
                            except Exception as e:
                                self.logger.log_error(f'Error cleaning input folder {path}: {str(e)}')
                    
                    if input_folders_cleaned > 0:
                        self.logger.log_system(f'Cleaned up {input_folders_cleaned} old input folders')

                # Clean up extract directories
                if os.path.exists(self.BASE_EXTRACT_FOLDER):
                    self.logger.log_system(f'Checking for old extract folders in {self.BASE_EXTRACT_FOLDER}')
                    extract_folders_cleaned = 0
                    for folder in os.listdir(self.BASE_EXTRACT_FOLDER):
                        if self.shutdown_flag.is_set():
                            break
                            
                        path = os.path.join(self.BASE_EXTRACT_FOLDER, folder)
                        if path in active_extract_folders:
                            continue
                            
                        if os.path.isdir(path):
                            try:
                                # Use modification time to determine age
                                mod_time = datetime.fromtimestamp(os.path.getmtime(path))
                                if current_time - mod_time > timedelta(hours=self.MAX_FILE_AGE_HOURS):
                                    shutil.rmtree(path)
                                    extract_folders_cleaned += 1
                                    self.logger.log_cleanup('old_extract', path)
                            except Exception as e:
                                self.logger.log_error(f'Error cleaning extract folder {path}: {str(e)}')
                    
                    if extract_folders_cleaned > 0:
                        self.logger.log_system(f'Cleaned up {extract_folders_cleaned} old extract folders')

                # Clean up output folders and ZIP files
                if os.path.exists(self.BASE_OUTPUT_FOLDER):
                    self.logger.log_system(f'Checking for old output folders and ZIP files in {self.BASE_OUTPUT_FOLDER}')
                    output_cleaned = 0
                    zip_cleaned = 0
                    
                    for item in os.listdir(self.BASE_OUTPUT_FOLDER):
                        if self.shutdown_flag.is_set():
                            break

                        path = os.path.join(self.BASE_OUTPUT_FOLDER, item)
                        if path in active_output_folders:
                            continue

                        try:
                            is_old = False
                            # First try to parse from item name (for folders and ZIP files)
                            try:
                                if item.startswith('result_'):
                                    parts = item[7:].split('_')  # Remove 'result_' prefix
                                    if len(parts) >= 2:
                                        timestamp = parts[0] + '_' + parts[1]
                                        folder_time = datetime.strptime(timestamp, '%Y%m%d_%H%M%S')
                                        if current_time - folder_time > timedelta(hours=self.MAX_OUTPUT_AGE_HOURS):
                                            is_old = True
                                else:
                                    # Try the old format too with session IDs
                                    timestamp = item.split('_')[0] + '_' + item.split('_')[1]
                                    folder_time = datetime.strptime(timestamp, '%Y%m%d_%H%M%S')
                                    if current_time - folder_time > timedelta(hours=self.MAX_OUTPUT_AGE_HOURS):
                                        is_old = True
                            except (ValueError, IndexError):
                                # If we can't parse from name, use modification time
                                mod_time = datetime.fromtimestamp(os.path.getmtime(path))
                                if current_time - mod_time > timedelta(hours=self.MAX_OUTPUT_AGE_HOURS):
                                    is_old = True
                            
                            if is_old:
                                if os.path.isdir(path):
                                    shutil.rmtree(path)
                                    output_cleaned += 1
                                    self.logger.log_cleanup('old_output_folder', path)
                                elif item.endswith('.zip'):
                                    os.remove(path)
                                    zip_cleaned += 1
                                    self.logger.log_cleanup('old_zip', path)
                        except Exception as e:
                            self.logger.log_error(f'Error cleaning output {path}: {str(e)}')
                    
                    if output_cleaned > 0 or zip_cleaned > 0:
                        self.logger.log_system(f'Cleaned up {output_cleaned} output folders and {zip_cleaned} ZIP files')

            except Exception as e:
                self.logger.log_error(f'Cleanup error: {str(e)}', details=traceback.format_exc())

            # Sleep for 30 minutes or until shutdown (using smaller increments to allow faster shutdown)
            self.logger.log_system('Cleanup complete, next run in 30 minutes')
            for _ in range(30):  # 30 minutes in 1-minute intervals
                if self.shutdown_flag.is_set():
                    break
                time.sleep(60)

        self.logger.log_system('File cleanup service stopped')

    def shutdown(self):
        """Shutdown the task handler gracefully."""
        self.logger.log_system('Shutting down task handler')
        self.shutdown_flag.set()

        # Wait for queue processor to stop
        while self.is_running:
            time.sleep(0.1)

        # Clean up thread pool
        self.thread_pool.shutdown(wait=True)
        self.logger.log_system('Task handler shutdown complete')

    def _update_queue_positions(self):
        """Update queue positions for all queued tasks."""
        queued_tasks = [(tid, task) for tid, task in self.active_tasks.items()
                        if task.get('status') == 'queued']
        for position, (tid, task) in enumerate(queued_tasks, 1):
            task['queue_position'] = position
            task['stage'] = f'Waiting in queue (position {position})'

    def _get_queue_position(self, task_id):
        """Get the position of a task in the queue (1-based)."""
        queued_tasks = [t for t in self.active_tasks.values() if t.get('status') == 'queued']
        for i, task in enumerate(queued_tasks, 1):
            if task.get('id') == task_id:
                return i
        return 0

    def update_task_status(self, task_id, status, output_folder=None, session_id=None, error_message=None, total_detections=None, has_detections=None):
        """Update task status in the task registry."""
        try:
            if task_id in self.active_tasks:
                task = self.active_tasks[task_id]
                task['status'] = status
                task['last_updated'] = time.time()
                
                if status == 'completed':
                    task['completed_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                if output_folder:
                    task['output_folder'] = output_folder
                    
                    # Look for the zip_path.txt file which contains the path to the output ZIP
                    zip_path_file = os.path.join(output_folder, "zip_path.txt")
                    if os.path.exists(zip_path_file):
                        with open(zip_path_file, "r") as f:
                            zip_path = f.read().strip()
                            if os.path.exists(zip_path):
                                task['zip_path'] = zip_path
                                self.logger.log_system(f"Found ZIP path from zip_path.txt: {zip_path}")
                    
                    # Also check for no_detections.txt 
                    no_detections_file = os.path.join(output_folder, "no_detections.txt")
                    if os.path.exists(no_detections_file):
                        task['has_detections'] = False
                        self.logger.log_system(f"Found no_detections.txt, setting has_detections=False")
                    else:
                        # Check if detections.json exists and if it contains detections
                        detections_file = os.path.join(output_folder, "detections.json")
                        if os.path.exists(detections_file):
                            try:
                                with open(detections_file, "r") as f:
                                    detections = json.load(f)
                                    task['has_detections'] = len(detections) > 0
                                    task['total_detections'] = sum(len(det.get("coordinates", [])) for det in detections)
                                    self.logger.log_system(f"Found {task['total_detections']} detections in {len(detections)} images")
                            except Exception as e:
                                self.logger.log_error(f"Error reading detections.json: {str(e)}")
                                task['has_detections'] = False
                
                if session_id:
                    task['session_id'] = session_id
                    
                if error_message:
                    task['error'] = error_message
                    
                if total_detections is not None:
                    task['total_detections'] = total_detections
                    
                if has_detections is not None:
                    task['has_detections'] = has_detections
                    
                self.logger.log_system(f"Updated task {task_id} status to {status}")
                self._save_tasks()
                return True
            else:
                self.logger.log_error(f"Task {task_id} not found for status update")
                return False
        except Exception as e:
            self.logger.log_error(f"Error updating task status: {str(e)}")
            return False

    def _save_tasks(self):
        """Save tasks to a persistent file."""
        try:
            # Create a serializable copy of the tasks
            serializable_tasks = {}
            
            for task_id, task in self.active_tasks.items():
                # Create a copy of the task without non-serializable items
                task_copy = task.copy()
                
                # Remove non-serializable items
                if 'future' in task_copy:
                    del task_copy['future']
                if 'extract_dir' in task_copy and isinstance(task_copy['extract_dir'], (set, dict)):
                    del task_copy['extract_dir']
                
                # Convert datetime objects to strings
                if 'created_at' in task_copy and isinstance(task_copy['created_at'], datetime):
                    task_copy['created_at'] = task_copy['created_at'].isoformat()
                
                serializable_tasks[task_id] = task_copy
            
            # Ensure the directory exists
            os.makedirs(os.path.dirname(self.TASKS_DB_FILE), exist_ok=True)
            
            # Save to file with pretty formatting
            with open(self.TASKS_DB_FILE, 'w') as f:
                json.dump(serializable_tasks, f, indent=2)
                
            self.logger.log_system(f"Saved {len(serializable_tasks)} tasks to {self.TASKS_DB_FILE}")
            
        except Exception as e:
            self.logger.log_error(f"Error saving tasks to file: {str(e)}")
            self.logger.log_error(traceback.format_exc())
    
    def _load_tasks(self):
        """Load tasks from persistent file."""
        try:
            if not os.path.exists(self.TASKS_DB_FILE):
                self.logger.log_system(f"Tasks database file not found: {self.TASKS_DB_FILE}")
                return
                
            if os.path.getsize(self.TASKS_DB_FILE) == 0:
                self.logger.log_system(f"Tasks database file is empty: {self.TASKS_DB_FILE}")
                return
                
            with open(self.TASKS_DB_FILE, 'r') as f:
                loaded_tasks = json.load(f)
                
            # Convert loaded tasks back to proper format
            for task_id, task in loaded_tasks.items():
                try:
                    # Convert ISO datetime strings back to datetime objects
                    if 'created_at' in task and isinstance(task['created_at'], str):
                        try:
                            task['created_at'] = datetime.fromisoformat(task['created_at'])
                        except ValueError:
                            task['created_at'] = datetime.now()
                    
                    # Add task to active_tasks
                    self.active_tasks[task_id] = task
                    
                    # Recreate cancellation events for any task
                    if task_id not in self.task_events:
                        self.task_events[task_id] = threading.Event()
                        
                    # Set cancellation flag if task was cancelled
                    if task.get('is_cancelled', False):
                        self.cancelled_tasks.add(task_id)
                        self.task_events[task_id].set()
                        
                    self.logger.log_system(f"Loaded task {task_id} ({task.get('status', 'unknown')})")
                except Exception as task_error:
                    self.logger.log_error(f"Error loading task {task_id}: {str(task_error)}")
            
            self.logger.log_system(f"Loaded {len(loaded_tasks)} tasks from {self.TASKS_DB_FILE}")
            
        except json.JSONDecodeError as json_error:
            self.logger.log_error(f"JSON decode error loading tasks from file: {str(json_error)}")
            self.logger.log_error(traceback.format_exc())
            # Backup corrupted file
            if os.path.exists(self.TASKS_DB_FILE):
                backup_path = f"{self.TASKS_DB_FILE}.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                try:
                    shutil.copy2(self.TASKS_DB_FILE, backup_path)
                    self.logger.log_system(f"Backed up corrupted tasks file to: {backup_path}")
                except Exception as backup_error:
                    self.logger.log_error(f"Error backing up corrupted tasks file: {str(backup_error)}")
        except Exception as e:
            self.logger.log_error(f"Error loading tasks from file: {str(e)}")
            self.logger.log_error(traceback.format_exc())
            # Start with empty tasks if there was an error loading