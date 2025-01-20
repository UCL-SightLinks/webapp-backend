"""Logger handler for API requests and important events."""

import os
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler

class LoggerHandler:
    """Handles logging of API requests and important events."""
    
    _instance = None  # Singleton instance
    
    def __new__(cls):
        """Ensure only one instance of LoggerHandler exists."""
        if cls._instance is None:
            cls._instance = super(LoggerHandler, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize logger with configuration."""
        # Only initialize once
        if self._initialized:
            return
            
        self.log_dir = 'logs'
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
        # Create logger
        self.logger = logging.getLogger('api_logger')
        
        # Clear any existing handlers
        if self.logger.handlers:
            self.logger.handlers.clear()
            
        self.logger.setLevel(logging.INFO)
        # Prevent propagation to root logger
        self.logger.propagate = False
        
        # Create rotating file handler (new file each day, keep 30 days of logs)
        log_file = os.path.join(self.log_dir, f'api_{datetime.now().strftime("%Y%m%d")}.log')
        handler = RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=30
        )
        
        # Create formatter
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(handler)
        
        self._initialized = True
    
    def log_request(self, method, endpoint, params=None, status_code=None, error=None):
        """Log an API request."""
        message = f"REQUEST: {method} {endpoint}"
        if params:
            message += f" | Params: {params}"
        if status_code:
            message += f" | Status: {status_code}"
        if error:
            message += f" | Error: {error}"
        self.logger.info(message)
    
    def log_task_status(self, task_id, status, progress=None, stage=None, error=None, **kwargs):
        """Log task status changes."""
        message = f"Task {task_id} status: {status}"
        details = {
            'task_id': task_id,
            'status': status,
            'progress': progress,
            'stage': stage,
            'error': error,
            **kwargs  # Include any additional parameters
        }
        # Filter out None values
        details = {k: v for k, v in details.items() if v is not None}
        self.logger.info(message)
    
    def log_file_operation(self, operation, filepath, success=True, error=None, details=None):
        """Log file operations.
        
        Args:
            operation (str): Type of operation (CREATE, SAVE, VERIFY, etc.)
            filepath (str): Path to the file
            success (bool): Whether the operation was successful
            error (str, optional): Error message if operation failed
            details (str, optional): Additional details about the operation
        """
        message = f"FILE {operation}: {filepath}"
        if not success:
            message += f" | Failed: {error}"
        if details:
            message += f" | {details}"
        self.logger.info(message)
    
    def log_error(self, error_message, details=None):
        """Log error messages."""
        message = f"ERROR: {error_message}"
        if details:
            message += f" | Details: {details}"
        self.logger.error(message)
    
    def log_cleanup(self, item_type, path):
        """Log cleanup operations."""
        self.logger.info(f"CLEANUP: Removed {item_type} at {path}")
    
    def log_system(self, message):
        """Log system-level messages."""
        self.logger.info(f"SYSTEM: {message}") 