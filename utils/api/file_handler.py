"""File handler module for managing file operations."""

import os
import uuid
from datetime import datetime
from werkzeug.utils import secure_filename
from flask import send_file, Response, send_from_directory
import zipfile
import traceback
from utils.api.logger_handler import LoggerHandler
from utils.api.request_handler import RequestHandler

class FileHandler:
    """Handles file operations including uploads and downloads."""
    
    def __init__(self):
        """Initialize file handler with configuration."""
        # Use absolute paths
        self.APP_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.BASE_UPLOAD_FOLDER = os.path.join(self.APP_ROOT, 'input')
        self.BASE_OUTPUT_FOLDER = os.path.join(self.APP_ROOT, 'run', 'output')
        self.ALLOWED_EXTENSIONS = {
            '0': {'zip', 'jpg', 'jpeg', 'png', 'jgw'},  # Type 0: jpg/png/jpeg + jgw or zip including these
            '1': {'zip', 'tif', 'tiff'}  # Type 1: GeoTIFF files or zip including them
        }
        
        # Create base directories if they don't exist
        for folder in [self.BASE_UPLOAD_FOLDER, self.BASE_OUTPUT_FOLDER]:
            if not os.path.exists(folder):
                os.makedirs(folder)
                print(f"Created directory: {folder}")
    
    def allowed_file(self, filename, input_type='0'):
        """Check if the file extension is allowed.
        
        Args:
            filename (str): The name of the file to check
            input_type (str): The input type ('0' for Digimap, '1' for custom)
            
        Returns:
            bool: True if the file extension is allowed, False otherwise
        """
        logger_handler = LoggerHandler()
        
        if '.' not in filename:
            logger_handler.log_system(f"File {filename} has no extension, rejected")
            return False
            
        ext = filename.rsplit('.', 1)[1].lower()
        allowed = self.ALLOWED_EXTENSIONS.get(input_type, self.ALLOWED_EXTENSIONS['0'])
        
        is_allowed = ext in allowed
        
        if is_allowed:
            logger_handler.log_system(f"File {filename} with extension {ext} is allowed for input_type {input_type}")
        else:
            logger_handler.log_system(f"File {filename} with extension {ext} is NOT allowed for input_type {input_type}")
            logger_handler.log_system(f"Allowed extensions for input_type {input_type}: {allowed}")
            
        return is_allowed
    
    def create_session_folders(self):
        """Create unique session folders for input and output.
        
        Returns:
            tuple: (session_id, input_folder_path)
        """
        session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        input_folder = os.path.join(self.BASE_UPLOAD_FOLDER, session_id)
        os.makedirs(input_folder, exist_ok=True)
        return session_id, input_folder
    
    def send_file_response(self, filepath):
        """Create a file response.

        Args:
            filepath (str): Path to the file to send.

        Returns:
            flask.Response: A response object with the file attached.
        """
        try:
            # Use the imported logger_handler
            logger_handler = LoggerHandler()
            
            logger_handler.log_system(f"Sending file: {filepath}")
            
            # Check if file exists
            if not os.path.exists(filepath):
                logger_handler.log_error(f"File does not exist: {filepath}")
                return {'error': f'File not found: {filepath}'}, 404
            
            # Get file size
            file_size = os.path.getsize(filepath)
            if file_size == 0:
                logger_handler.log_error(f"File is empty: {filepath}")
                return {'error': f'File is empty: {filepath}'}, 500
            
            # Get file extension to determine mime type
            _, ext = os.path.splitext(filepath)
            
            # Set MIME type based on extension
            mime_type = 'application/octet-stream'  # Default
            if ext.lower() == '.zip':
                mime_type = 'application/zip'
            elif ext.lower() == '.json':
                mime_type = 'application/json'
            elif ext.lower() == '.txt':
                mime_type = 'text/plain'
            
            # Fixed filename for consistency
            filename = "results" + ext
            
            # Use Flask's send_file directly
            logger_handler.log_debug(f"Sending file {filepath} as {filename} with MIME type {mime_type}")
            return send_file(
                filepath,
                mimetype=mime_type,
                as_attachment=True,
                download_name=filename
            )
        except Exception as e:
            logger_handler.log_error(f"Error sending file: {str(e)}", details=traceback.format_exc())
            return {'error': f'Error sending file: {str(e)}'}, 500 