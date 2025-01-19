"""File handler module for managing file operations."""

import os
import uuid
from datetime import datetime
from werkzeug.utils import secure_filename
from flask import send_file, Response, send_from_directory
import zipfile
import traceback

class FileHandler:
    """Handles file operations including uploads and downloads."""
    
    def __init__(self):
        """Initialize file handler with configuration."""
        self.BASE_UPLOAD_FOLDER = 'input'
        self.BASE_OUTPUT_FOLDER = 'run/output'
        self.ALLOWED_EXTENSIONS = {
            '0': {'zip'},  # Digimap data - only zip files
            '1': {'zip', 'jpg', 'jgw'}  # Custom data - zip, jpg, and jgw files
        }
        
        # Create base directories if they don't exist
        for folder in [self.BASE_UPLOAD_FOLDER, self.BASE_OUTPUT_FOLDER]:
            if not os.path.exists(folder):
                os.makedirs(folder)
    
    def allowed_file(self, filename, input_type='0'):
        """Check if the file extension is allowed.
        
        Args:
            filename (str): The name of the file to check
            input_type (str): The input type ('0' for Digimap, '1' for custom)
            
        Returns:
            bool: True if the file extension is allowed, False otherwise
        """
        if '.' not in filename:
            return False
            
        ext = filename.rsplit('.', 1)[1].lower()
        allowed = self.ALLOWED_EXTENSIONS.get(input_type, self.ALLOWED_EXTENSIONS['0'])
        return ext in allowed
    
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
        """Send a file as a response.
        
        Args:
            filepath (str): Path to the file to send
            
        Returns:
            Response: Flask response object
        """
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"File not found: {filepath}")
                
            file_size = os.path.getsize(filepath)
            if file_size == 0:
                raise ValueError("File is empty")
                
            response = send_file(
                filepath,
                as_attachment=True,
                download_name=os.path.basename(filepath),
                mimetype='application/zip'
            )
            response.headers['Content-Length'] = file_size
            response.headers['Content-Type'] = 'application/zip'
            response.headers['Content-Disposition'] = f'attachment; filename={os.path.basename(filepath)}'
            return response
            
        except Exception as e:
            print(f"Error sending file: {str(e)}")
            print("Traceback:", traceback.format_exc())
            raise 