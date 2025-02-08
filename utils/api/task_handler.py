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
import psutil

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

        # Initialize logger
        self.logger = LoggerHandler()

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
                'error': 'No task ID provided'
            }

        with self.task_lock:
            task = self.active_tasks.get(task_id)
            if not task:
                return {
                    'status': 'unknown',
                    'error': f'Task {task_id} not found'
                }

            # Add queue position for queued tasks
            if task.get('status') == 'queued':
                queue_position = self._get_queue_position(task_id)
                task = task.copy()  # Create a copy to modify
                task['queue_position'] = queue_position
                task['stage'] = f'Waiting in queue (position {queue_position})'

            return task.copy()  # Return a copy to prevent modification

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
                        future.cancel()
                        try:
                            future.result(timeout=1.0)
                        except:
                            pass
                    del task['future']

            # Clean up input folder if exists
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
        with self.task_lock:
            if task_id in self.active_tasks:
                return self.active_tasks[task_id].get('is_cancelled', False)
            return task_id in self.cancelled_tasks

    def process_task(self, task_id, input_folder, params, execute_func):
        """Process a task directly (synchronously)."""
        try:
            # Initialize task data
            if not task_id:
                task_id = str(uuid.uuid4())

            session_id = os.path.basename(input_folder)

            with self.task_lock:
                # Check if task already exists
                existing_task = self.active_tasks.get(task_id)
                if existing_task:
                    task_data = existing_task.copy()
                    task_data.update({
                        'status': 'processing',
                        'progress': 0,
                        'stage': 'Initializing',
                        'session_id': session_id  # Ensure session_id is preserved
                    })
                else:
                    task_data = {
                        'id': task_id,
                        'status': 'processing',
                        'progress': 0,
                        'stage': 'Initializing',
                        'created_at': datetime.now(),
                        'input_folder': input_folder,
                        'session_id': session_id,
                        'is_cancelled': False
                    }

                # Update task in active_tasks
                self.active_tasks[task_id] = task_data

                # Create cancellation event if needed
                if task_id not in self.task_events:
                    self.task_events[task_id] = threading.Event()

                # Update processing stats
                with self.stats_lock:
                    self.stats['current_tasks'] = len([t for t in self.active_tasks.values()
                                                     if t.get('status') == 'processing'])

            self.logger.log_task_status(task_id, 'processing')

            try:
                # Process the task
                output_folder = self._execute_task(task_id, input_folder, params, execute_func)

                if not output_folder or not os.path.exists(output_folder):
                    raise Exception("No output folder was generated")

                # Check if task was cancelled during execution
                with self.task_lock:
                    current_task = self.active_tasks.get(task_id)
                    if not current_task:
                        raise Exception("Task state lost during execution")
                    if current_task.get('is_cancelled', False):
                        raise Exception("Task was cancelled during execution")

                # Create ZIP file
                self.logger.log_system(f"Creating ZIP file for task {task_id}")
                zip_path = self._create_zip_file(task_id, output_folder)

                if not zip_path or not os.path.exists(zip_path):
                    raise Exception("Failed to create ZIP file")

                # Update task status
                with self.task_lock:
                    current_task = self.active_tasks.get(task_id)
                    if current_task and not current_task.get('is_cancelled', False):
                        current_task.update({
                            'status': 'completed',
                            'progress': 100,
                            'stage': 'Completed',
                            'zip_path': zip_path,
                            'session_id': session_id  # Ensure session_id is preserved
                        })
                        self.active_tasks[task_id] = current_task

                return task_id

            except Exception as e:
                error_msg = str(e)
                self.logger.log_error(f"Task processing failed: {error_msg}")

                # Update task status
                with self.task_lock:
                    current_task = self.active_tasks.get(task_id)
                    if current_task:
                        if current_task.get('is_cancelled', False):
                            current_task.update({
                                'status': 'cancelled',
                                'error': 'Task cancelled by user',
                                'stage': 'Cancelled',
                                'session_id': session_id  # Ensure session_id is preserved
                            })
                        else:
                            current_task.update({
                                'status': 'failed',
                                'error': error_msg,
                                'stage': 'Failed',
                                'session_id': session_id  # Ensure session_id is preserved
                            })
                        self.active_tasks[task_id] = current_task
                raise

        except Exception as e:
            with self.stats_lock:
                self.stats['current_tasks'] = len([t for t in self.active_tasks.values()
                                                 if t.get('status') == 'processing'])
                self.stats['failed_tasks'] += 1

            self.logger.log_error(f"Error in process_task: {str(e)}")

            # Ensure task exists and is marked as failed
            with self.task_lock:
                if task_id:
                    self.active_tasks[task_id] = {
                        'id': task_id,
                        'status': 'failed',
                        'error': str(e),
                        'stage': 'Failed',
                        'created_at': datetime.now(),
                        'session_id': session_id,  # Ensure session_id is preserved
                        'is_cancelled': False
                    }
            return task_id

    def _execute_task(self, task_id, input_folder, params, execute_func):
        """Execute the task with proper progress tracking and cancellation checks."""
        try:
            def progress_callback(stage, progress):
                if task_id in self.active_tasks:
                    self.active_tasks[task_id].update({
                        'progress': progress,
                        'stage': stage
                    })

            # Check for cancellation before starting execution
            if self.check_cancellation(task_id):
                raise Exception("Task cancelled by user")

            # Execute the task
            output_folder = execute_func(
                input_folder,
                params['input_type'],
                params['classification_threshold'],
                params['prediction_threshold'],
                params['save_labeled_image'],
                params['output_type'],
                params['yolo_model_type'],
                progress_callback,
                lambda: self.check_cancellation(task_id)
            )

            return output_folder

        except Exception as e:
            self.logger.log_error(f"Task execution failed: {str(e)}")
            raise

    def _create_zip_file(self, task_id, output_folder):
        """Create a ZIP file for the task."""
        try:
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

                    # Clean up old tasks from active_tasks ONLY if they are in a final state
                    current_tasks = list(self.active_tasks.items())
                    for task_id, task in current_tasks:
                        # Only clean up tasks that are in a final state and older than the threshold
                        if task.get('status') in ['completed', 'failed', 'cancelled']:
                            created_at = task.get('created_at')
                            if created_at and (current_time - created_at) > timedelta(hours=self.MAX_OUTPUT_AGE_HOURS):
                                # Double check task is not processing or queued
                                if task_id not in self.processing_tasks and task.get('status') not in ['queued', 'processing']:
                                    del self.active_tasks[task_id]
                                    if task_id in self.cancelled_tasks:
                                        self.cancelled_tasks.remove(task_id)
                                    if task_id in self.task_events:
                                        del self.task_events[task_id]
                                    self.logger.log_cleanup('task_data', f'Task {task_id}')

                # Clean up input folders
                if os.path.exists(self.BASE_UPLOAD_FOLDER):
                    for folder in os.listdir(self.BASE_UPLOAD_FOLDER):
                        if self.shutdown_flag.is_set():
                            break

                        path = os.path.join(self.BASE_UPLOAD_FOLDER, folder)
                        if path in active_input_folders:
                            continue

                        if os.path.isdir(path):
                            try:
                                folder_time = datetime.strptime(folder.split('_')[0] + '_' + folder.split('_')[1], '%Y%m%d_%H%M%S')
                                if current_time - folder_time > timedelta(hours=self.MAX_FILE_AGE_HOURS):
                                    shutil.rmtree(path)
                                    self.logger.log_cleanup('old_input', path)
                            except (ValueError, IndexError):
                                continue

                # Clean up output folders and ZIP files
                if os.path.exists(self.BASE_OUTPUT_FOLDER):
                    for item in os.listdir(self.BASE_OUTPUT_FOLDER):
                        if self.shutdown_flag.is_set():
                            break

                        path = os.path.join(self.BASE_OUTPUT_FOLDER, item)
                        if path in active_output_folders:
                            continue

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

            # Sleep for 30 minutes or until shutdown
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