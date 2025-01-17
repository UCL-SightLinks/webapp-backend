import os
import zipfile

def compress_folder_to_zip(folder_path, zip_file_name="results.zip"):
    """
    Compress a folder and its contents into a ZIP file.

    Args:
        folder_path (str): Path to the folder to compress.
        zip_file_name (str): Name of the resulting ZIP file (include .zip extension).

    Returns:
        str: Path to the created ZIP file on success, error message on failure.
    """
    zip_file_path = os.path.join(folder_path, zip_file_name)
    
    try:
        # Validate folder path
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"The folder {folder_path} does not exist")
            
        # Create the ZIP file
        with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(folder_path):
                for file in files:
                    # Skip the zip file itself if it exists
                    if file == zip_file_name:
                        continue
                        
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, start=folder_path)
                    
                    try:
                        zipf.write(file_path, arcname)
                    except Exception as e:
                        return f"Failed to add file {file_path}: {str(e)}"
        
        # Verify the zip file was created
        if not os.path.exists(zip_file_path):
            raise FileNotFoundError("ZIP file was not created")
            
        return os.path.abspath(zip_file_path)
        
    except Exception as e:
        return f"Compression failed: {str(e)}"
