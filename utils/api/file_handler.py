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
        self.ALLOWED_EXTENSIONS = {'zip'}
        
        # Create base directories if they don't exist
        for folder in [self.BASE_UPLOAD_FOLDER, self.BASE_OUTPUT_FOLDER]:
            if not os.path.exists(folder):
                os.makedirs(folder)
    
    def allowed_file(self, filename):
        """Check if the file extension is allowed.
        
        Args:
            filename (str): The name of the file to check
            
        Returns:
            bool: True if the file extension is allowed, False otherwise
        """
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in self.ALLOWED_EXTENSIONS
    
    def create_session_folders(self):
        """Create unique session folders for input and output.
        
        Returns:
            tuple: (session_id, input_folder_path)
        """
        session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        input_folder = os.path.join(self.BASE_UPLOAD_FOLDER, session_id)
        os.makedirs(input_folder, exist_ok=True)
        return session_id, input_folder
    
    def send_file_response(self, file_path):
        """Helper function to handle send_file with version compatibility.
        
        Args:
            file_path (str): Path to the file to send
            
        Returns:
            Response: Flask response object with the file
            
        Raises:
            Exception: If file does not exist or is invalid
        """
        print(f"\n=== Sending File Response ===")
        print(f"File path: {file_path}")
        
        # Verify file exists and is valid
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            raise Exception(f"File not found: {file_path}")
            
        if not os.path.isfile(file_path):
            print(f"Path is not a file: {file_path}")
            raise Exception(f"Path is not a file: {file_path}")
            
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            print("File is empty")
            raise Exception("File is empty")
            
        print(f"File exists: Yes")
        print(f"File size: {file_size} bytes")
        
        try:
            # Verify zip file integrity
            print("\nVerifying ZIP file...")
            with zipfile.ZipFile(file_path, 'r') as zf:
                contents = zf.namelist()
                print(f"ZIP contents ({len(contents)} files):")
                for item in contents:
                    info = zf.getinfo(item)
                    print(f"- {item} (size: {info.file_size} bytes, compressed: {info.compress_size} bytes)")
                
                test_result = zf.testzip()
                if test_result is not None:
                    raise Exception(f"ZIP file is corrupted at: {test_result}")
                    
                if not contents:
                    raise Exception("ZIP file is empty")
            
            print("\nZIP file verified successfully")
            
            # Get directory and filename
            directory = os.path.dirname(file_path)
            filename = os.path.basename(file_path)
            print(f"Directory: {directory}")
            print(f"Filename: {filename}")
            
            # Use send_file with explicit binary mode
            response = send_file(
                file_path,
                as_attachment=True,
                download_name='result.zip',
                mimetype='application/zip'
            )
            
            # Set additional headers
            response.headers['Content-Length'] = file_size
            response.headers['Content-Type'] = 'application/zip'
            response.headers['Cache-Control'] = 'no-cache'
            
            print("\nFile response created successfully")
            print(f"Content-Length: {response.headers.get('Content-Length')}")
            print(f"Content-Type: {response.headers.get('Content-Type')}")
            print("=== Send File Response Complete ===\n")
            
            return response
            
        except Exception as e:
            print(f"Error sending file: {str(e)}")
            print("Traceback:", traceback.format_exc())
            raise Exception(f"Failed to send file: {str(e)}") 