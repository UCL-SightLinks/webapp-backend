"""Request handler module for managing API requests."""

import os
from flask import jsonify
from werkzeug.utils import secure_filename
from utils.api.logger_handler import LoggerHandler

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
        logger_handler = LoggerHandler()
        
        print("\n=== DEBUG: New Request ===")
        print(f"Content-Type: {request.content_type}")
        print(f"Accept: {request.headers.get('Accept')}")
        
        print("\nDEBUG: Received Files:")
        print(f"request.files keys: {list(request.files.keys())}")
        
        # Create session folders first
        session_id, input_folder = self.file_handler.create_session_folders()
        logger_handler.log_system(f"Created session folders - ID: {session_id}, Input folder: {input_folder}")
        
        # Get input type first to validate file extensions
        input_type = request.form.get('input_type', '0')
        logger_handler.log_system(f"Input Type: {input_type}")
        logger_handler.log_system(f"Allowed Extensions: {self.file_handler.ALLOWED_EXTENSIONS.get(input_type)}")
        
        # Get a complete list of all files from all keys to handle multiple file inputs correctly
        all_uploaded_files = []
        for key in request.files:
            files_for_key = request.files.getlist(key)
            print(f"DEBUG: Key '{key}' has {len(files_for_key)} file(s)")
            for file in files_for_key:
                if file and file.filename:
                    all_uploaded_files.append(file)
                    print(f"DEBUG: File from key '{key}': {file.filename}")
                    if '.' in file.filename:
                        ext = file.filename.rsplit('.', 1)[1].lower()
                        print(f"  Extension: {ext}")
                else:
                    print(f"DEBUG: Empty file for key '{key}'")
                    
        print(f"DEBUG: Total files received: {len(all_uploaded_files)}")
        print(f"DEBUG: Form Data: {request.form.to_dict()}")
        print(f"DEBUG: Is JSON? {request.is_json}")
        
        # Process files based on input type
        files = []
        jpg_jpeg_png_found = False
        jgw_found = False
        tif_found = False
        zip_found = False
        
        logger_handler.log_system("\nValidating uploaded files:")
        for file in all_uploaded_files:
            if not file.filename:
                continue
                
            logger_handler.log_system(f"Validating file: {file.filename}")
            
            if not self.file_handler.allowed_file(file.filename, input_type):
                logger_handler.log_system(f"File type not allowed for input_type {input_type}: {file.filename}")
                continue
            
            ext = file.filename.rsplit('.', 1)[1].lower()
            
            # Track file types based on extension
            if ext == 'zip':
                logger_handler.log_system("ZIP file found")
                zip_found = True
                files.append(file)
            elif ext in ['tif', 'tiff'] and input_type == '1':
                logger_handler.log_system("TIF/TIFF file found")
                tif_found = True
                
                # Save the file first
                temp_path = os.path.join(input_folder, secure_filename(file.filename))
                file.save(temp_path)
                logger_handler.log_system(f"TIF file saved to: {temp_path}")
                
                # Verify file was saved properly
                if not os.path.exists(temp_path):
                    raise ValueError(f"Failed to save TIF file: {temp_path}")
                    
                file_size = os.path.getsize(temp_path)
                if file_size == 0:
                    raise ValueError(f"TIF file was saved with 0 bytes: {temp_path}")
                    
                logger_handler.log_system(f"TIF file size: {file_size} bytes")
                
                # Log TIF file details
                try:
                    import rasterio
                    with rasterio.open(temp_path) as src:
                        logger_handler.log_system(f"TIF file details:")
                        logger_handler.log_system(f"- Width: {src.width}")
                        logger_handler.log_system(f"- Height: {src.height}")
                        logger_handler.log_system(f"- Bands: {src.count}")
                        logger_handler.log_system(f"- CRS: {src.crs}")
                        logger_handler.log_system(f"- Transform: {src.transform}")
                        
                        # Validate TIF file
                        if src.count not in [1, 3, 4]:
                            logger_handler.log_error(f"Invalid number of bands in TIF file: {src.count}")
                            raise ValueError(f"TIF file must have 1, 3, or 4 bands, got {src.count}")
                            
                        if src.width == 0 or src.height == 0:
                            logger_handler.log_error("TIF file has zero dimensions")
                            raise ValueError("TIF file has zero dimensions")
                            
                except Exception as e:
                    logger_handler.log_error(f"Error validating TIF file: {str(e)}")
                    # Clean up the invalid file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    raise ValueError(f"Invalid TIF file: {str(e)}")
                
                # Reset file position to beginning
                file.seek(0)
                
                # Add the file object to the list for processing
                files.append(file)
                logger_handler.log_system(f"TIF file added to processing queue: {file.filename}")
            elif ext in ['jpg', 'jpeg', 'png'] and input_type == '0':
                logger_handler.log_system("JPG/JPEG/PNG file found")
                jpg_jpeg_png_found = True
                files.append(file)
            elif ext == 'jgw' and input_type == '0':
                logger_handler.log_system("JGW file found")
                jgw_found = True
                files.append(file)
            else:
                logger_handler.log_system(f"Skipping file: {file.filename} - not valid for input_type {input_type}")
        
        # Create validation summary
        logger_handler.log_system(f"\nValidation summary:")
        logger_handler.log_system(f"ZIP found: {zip_found}")
        logger_handler.log_system(f"JPG/JPEG/PNG found: {jpg_jpeg_png_found}")
        logger_handler.log_system(f"JGW found: {jgw_found}")
        logger_handler.log_system(f"TIF/TIFF found: {tif_found}")
        logger_handler.log_system(f"Total valid files: {len(files)}")
        
        # Modified validation logic to handle multiple files
        if input_type == '0':
            if not zip_found and not jpg_jpeg_png_found:
                raise ValueError("No valid image files uploaded. Expected JPG/JPEG/PNG files or ZIP containing them.")
        elif input_type == '1':
            if not zip_found and not tif_found:
                raise ValueError("No valid files uploaded - expected ZIP or TIF/TIFF files")
        
        logger_handler.log_system(f"Validation passed - processing {len(files)} files")
        
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
                'yolo_model_type': str(json_data.get('yolo_model_type', 'm'))
            }
        else:
            params = {
                'input_type': input_type,
                'classification_threshold': request.form.get('classification_threshold', '0.35'),
                'prediction_threshold': request.form.get('prediction_threshold', '0.5'),
                'save_labeled_image': request.form.get('save_labeled_image', 'false'),
                'output_type': request.form.get('output_type', '0'),
                'yolo_model_type': request.form.get('yolo_model_type', 'm')
            }
        
        logger_handler.log_system(f"Content Type: {request.content_type}")
        logger_handler.log_system(f"Processed Parameters: {params}")
        logger_handler.log_system(f"Parameter Types: {str({k: type(v).__name__ for k, v in params.items()})}")
        
        # Make output type clearer in logs
        output_type = int(str(params.get('output_type', '0')))
        isTXT = output_type == 1
        logger_handler.log_system("\nOutput Type Analysis:")
        logger_handler.log_system(f"Raw output_type parameter: {params.get('output_type')}")
        logger_handler.log_system(f"Converted output_type: {output_type}")
        logger_handler.log_system(f"isTXT: {isTXT}")
        logger_handler.log_system(f"Will generate: {'TXT' if isTXT else 'JSON'}")
        logger_handler.log_system("=================\n")
        
        return files, params
    
    def create_error_response(self, message, status_code=400):
        """Create an error response."""
        return {'error': message}, status_code
    
    def create_success_response(self, data):
        """Create a success response."""
        return data, 200
    
    def save_uploaded_files(self, files, input_folder):
        """Save uploaded files to the input folder."""
        logger_handler = LoggerHandler()
        saved_files = []
        
        logger_handler.log_system("\n=== Saving Uploaded Files ===")
        logger_handler.log_system(f"Input folder: {input_folder}")
        logger_handler.log_system(f"Number of files to save: {len(files)}")
        
        for file in files:
            filename = secure_filename(file.filename)
            filepath = os.path.join(input_folder, filename)
            logger_handler.log_system(f"\nSaving file: {filename}")
            logger_handler.log_system(f"Full path: {filepath}")
            
            # Check if file already exists (was saved during validation)
            if os.path.exists(filepath):
                size = os.path.getsize(filepath)
                logger_handler.log_system(f"File already exists. Size: {size} bytes")
                saved_files.append(filepath)
                continue
            
            # Reset file position to beginning before saving
            file.seek(0)
            
            # Save file if it doesn't exist
            file.save(filepath)
            
            # Verify file was saved
            if os.path.exists(filepath):
                size = os.path.getsize(filepath)
                logger_handler.log_system(f"File saved successfully. Size: {size} bytes")
            else:
                logger_handler.log_error(f"File was not saved successfully: {filepath}")
            
            saved_files.append(filepath)
            
        logger_handler.log_system("\n=== Saved Files Summary ===")
        logger_handler.log_system(f"Total files saved: {len(saved_files)}")
        for path in saved_files:
            logger_handler.log_system(f"- {path}")
        logger_handler.log_system("=======================\n")
        
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