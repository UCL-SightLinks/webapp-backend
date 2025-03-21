"""Request handler module for managing API requests."""

import os
from flask import jsonify
from werkzeug.utils import secure_filename

class RequestHandler:
    """Handles API request parsing and response creation."""
    
    def __init__(self, file_handler):
        """Initialize request handler with file handler."""
        self.file_handler = file_handler
    
    def parse_request_parameters(self, request):
        """Parse and validate request parameters.
        
        Args:
            request: Flask request object
            
        Returns:
            tuple: (files_list, params)
            
        Raises:
            ValueError: If request is invalid
        """
        print("\n=== New Request ===")
        print("Content-Type:", request.content_type)
        print("Accept:", request.headers.get('Accept'))
        print("\nReceived Files:")
        for key in request.files:
            file = request.files[key]
            if file and file.filename:
                print(f"- File key '{key}': {file.filename}")
            else:
                print(f"- Empty file for key '{key}'")
        print("\nForm Data:", request.form.to_dict())
        print("Raw Data:", request.get_data())
        print("Is JSON?", request.is_json)
        print("JSON Data:", request.get_json(silent=True))
        print("Headers:", dict(request.headers))
        print("=================\n")
        
        # Get input type first to validate file extensions
        input_type = request.form.get('input_type', '0')
        
        # Handle file uploads based on input type
        if input_type == '0':
            # For Digimap data, expect single ZIP file
            if 'file' not in request.files:
                raise ValueError('No file uploaded')
                
            file = request.files['file']
            if not file or not file.filename:
                raise ValueError('No file selected')
                
            if not self.file_handler.allowed_file(file.filename, input_type):
                raise ValueError('Invalid file type - expected ZIP file')
                
            files = [file]
            
        else:
            # For custom data, handle either ZIP file or JPG+JGW pairs
            files = []
            jpg_found = False
            jgw_found = False
            zip_found = False
            
            # Check all uploaded files
            print("\nProcessing uploaded files:")
            for key in request.files:
                file = request.files[key]
                if not file or not file.filename:
                    continue
                
                print(f"Processing file: {file.filename}")
                
                if not self.file_handler.allowed_file(file.filename, input_type):
                    raise ValueError(f'Invalid file type: {file.filename}')
                
                ext = file.filename.rsplit('.', 1)[1].lower()
                print(f"File extension: {ext}")
                
                if ext == 'zip':
                    print("ZIP file found")
                    zip_found = True
                    files = [file]  # If ZIP file found, use only that
                    break  # No need to check other files
                elif ext == 'jpg':
                    print("JPG file found")
                    jpg_found = True
                    files.append(file)
                elif ext == 'jgw':
                    print("JGW file found")
                    jgw_found = True
                    files.append(file)
            
            print(f"\nValidation summary:")
            print(f"ZIP found: {zip_found}")
            print(f"JPG found: {jpg_found}")
            print(f"JGW found: {jgw_found}")
            print(f"Total files to process: {len(files)}")
            
            # Validate we have either a ZIP file or both JPG and JGW files
            if not zip_found and not (jpg_found and jgw_found):
                error_msg = []
                if not zip_found and not jpg_found and not jgw_found:
                    error_msg.append("No valid files uploaded")
                elif not jpg_found:
                    error_msg.append("JPG file is missing")
                elif not jgw_found:
                    error_msg.append("JGW file is missing")
                raise ValueError(f"Upload error: {', '.join(error_msg)}")
        
        # Get parameters based on content type
        if request.content_type == 'application/json':
            json_data = request.get_json()
            if not json_data:
                raise ValueError('Invalid JSON data')
            
            params = {
                'input_type': input_type,
                'classification_threshold': str(json_data.get('classification_threshold', '0.35')),
                'prediction_threshold': str(json_data.get('prediction_threshold', '0.5')),
                'save_labeled_image': str(json_data.get('save_labeled_image', 'false')),
                'output_type': str(json_data.get('output_type', '0')),
                'yolo_model_type': str(json_data.get('yolo_model_type', 'n'))
            }
        else:
            params = {
                'input_type': input_type,
                'classification_threshold': request.form.get('classification_threshold', '0.35'),
                'prediction_threshold': request.form.get('prediction_threshold', '0.5'),
                'save_labeled_image': request.form.get('save_labeled_image', 'false'),
                'output_type': request.form.get('output_type', '0'),
                'yolo_model_type': request.form.get('yolo_model_type', 'n')
            }
        
        print("Content Type:", request.content_type)
        print("Processed Parameters:", params)
        print("Parameter Types:", {k: type(v).__name__ for k, v in params.items()})
        
        # Make output type clearer in logs
        output_type = int(str(params.get('output_type', '0')))
        isTXT = output_type == 1
        print("\nOutput Type Analysis:")
        print(f"Raw output_type parameter: {params.get('output_type')}")
        print(f"Converted output_type: {output_type}")
        print(f"isTXT: {isTXT}")
        print(f"Will generate: {'TXT' if isTXT else 'JSON'}")
        print("=================\n")
        
        return files, params
    
    def create_error_response(self, message, status_code=400):
        """Create an error response."""
        return {'error': message}, status_code
    
    def create_success_response(self, data):
        """Create a success response."""
        return data, 200
    
    def save_uploaded_files(self, files, input_folder):
        """Save uploaded files to the input folder.
        
        Args:
            files: List of file objects from request
            input_folder (str): Path to input folder
            
        Returns:
            list: Paths to saved files
        """
        saved_files = []
        for file in files:
            filename = secure_filename(file.filename)
            filepath = os.path.join(input_folder, filename)
            file.save(filepath)
            saved_files.append(filepath)
        return saved_files
    
    def wants_json_response(self, request):
        """Check if the client wants a JSON response."""
        # For download endpoint, always return file
        if request.endpoint == 'download_result':
            return False
            
        # For other endpoints, check if client wants JSON
        wants_json = (
            request.content_type == 'application/json' or
            'application/json' in request.accept_mimetypes or
            request.headers.get('Accept') == 'application/json'
        )
        return wants_json 