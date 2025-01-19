"""Request handler module for managing API requests."""

import os
from flask import jsonify
from werkzeug.utils import secure_filename

class RequestHandler:
    """Handles API request processing and parameter validation."""
    
    def __init__(self, file_handler):
        """Initialize request handler with dependencies.
        
        Args:
            file_handler (FileHandler): Instance of FileHandler for file operations
        """
        self.file_handler = file_handler
    
    def parse_request_parameters(self, request):
        """Parse and validate request parameters.
        
        Args:
            request: Flask request object
            
        Returns:
            tuple: (file_object, parameters_dict)
            
        Raises:
            ValueError: If request is invalid
        """
        print("\n=== New Request ===")
        print("Content-Type:", request.content_type)
        print("Accept:", request.headers.get('Accept'))
        print("Files:", request.files.keys())
        print("Form Data:", request.form.to_dict())
        print("Raw Data:", request.get_data())
        print("Is JSON?", request.is_json)
        print("JSON Data:", request.get_json(silent=True))
        print("Headers:", dict(request.headers))
        print("=================\n")
        
        # Validate file upload
        if 'file' not in request.files:
            raise ValueError('No file uploaded')
        
        file = request.files['file']
        if file.filename == '' or not self.file_handler.allowed_file(file.filename):
            raise ValueError('Invalid file')
        
        # Get parameters based on content type
        if request.content_type == 'application/json':
            json_data = request.get_json()
            if not json_data:
                raise ValueError('Invalid JSON data')
            
            params = {
                'input_type': str(json_data.get('input_type', '0')),
                'classification_threshold': str(json_data.get('classification_threshold', '0.35')),
                'prediction_threshold': str(json_data.get('prediction_threshold', '0.5')),
                'save_labeled_image': str(json_data.get('save_labeled_image', 'false')),
                'output_type': str(json_data.get('output_type', '0')),  # Default to JSON (0)
                'yolo_model_type': str(json_data.get('yolo_model_type', 'n'))
            }
        else:
            params = {
                'input_type': request.form.get('input_type', '0'),
                'classification_threshold': request.form.get('classification_threshold', '0.35'),
                'prediction_threshold': request.form.get('prediction_threshold', '0.5'),
                'save_labeled_image': request.form.get('save_labeled_image', 'false'),
                'output_type': request.form.get('output_type', '0'),  # Default to JSON (0)
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
        
        return file, params
    
    def save_uploaded_file(self, file, input_folder):
        """Save uploaded file to the input folder.
        
        Args:
            file: File object from request
            input_folder (str): Path to input folder
            
        Returns:
            str: Path to saved file
        """
        filename = secure_filename(file.filename)
        filepath = os.path.join(input_folder, filename)
        file.save(filepath)
        return filepath
    
    def wants_json_response(self, request):
        """Check if the client wants a JSON response.
        
        Args:
            request: Flask request object
            
        Returns:
            bool: True if client wants JSON, False otherwise
        """
        print("\n=== Response Decision ===")
        print(f"Content-Type: {request.content_type}")
        print(f"Accept Header: {request.headers.get('Accept')}")
        print(f"Accept Mimetypes: {request.accept_mimetypes}")
        print(f"Best Match: {request.accept_mimetypes.best}")
        
        # For download endpoint, always return file
        if request.endpoint == 'download_result':
            print("Endpoint is download_result, forcing file response")
            print("======================\n")
            return False
            
        # For other endpoints, check if client wants JSON
        wants_json = (
            request.content_type == 'application/json' or
            'application/json' in request.accept_mimetypes or
            request.headers.get('Accept') == 'application/json'
        )
        print(f"Wants JSON? {wants_json}")
        print("======================\n")
        return wants_json
    
    def create_error_response(self, message, status_code):
        """Create a standardized error response.
        
        Args:
            message (str): Error message
            status_code (int): HTTP status code
            
        Returns:
            tuple: (response_json, status_code)
        """
        return jsonify({'error': message}), status_code
    
    def create_success_response(self, data):
        """Create a standardized success response.
        
        Args:
            data (dict): Response data
            
        Returns:
            Response: JSON response object
        """
        return jsonify(data) 