import os
import zipfile
import shutil

def compress_folder_to_zip(folder_path, zip_file_name="results.zip"):
    """
    Compress a folder's contents into a ZIP file and delete the original folder.
    Places the ZIP file in the same directory as the folder.

    Args:
        folder_path (str): Path to the folder to compress.
        zip_file_name (str): Name of the resulting ZIP file (include .zip extension).

    Returns:
        str: Path to the created ZIP file.
        
    Raises:
        FileNotFoundError: If the folder doesn't exist or ZIP file creation fails.
        Exception: For other compression errors.
    """
    # Create ZIP file in the same directory as the folder
    zip_file_path = os.path.join(os.path.dirname(folder_path), zip_file_name)
    
    # Validate folder path
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"The folder {folder_path} does not exist")
        
    try:
        # Create the ZIP file
        with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    # Create archive name relative to the folder being compressed
                    arcname = os.path.relpath(file_path, folder_path)
                    zipf.write(file_path, arcname)
        
        # Verify the zip file was created and has content
        if not os.path.exists(zip_file_path) or os.path.getsize(zip_file_path) == 0:
            raise FileNotFoundError("ZIP file was not created or is empty")
            
        # Verify the ZIP file is valid
        with zipfile.ZipFile(zip_file_path) as zf:
            if zf.testzip() is not None:
                raise Exception("ZIP file is corrupted")
        
        # Delete the original folder after successful compression
        shutil.rmtree(folder_path)
        print(f"Deleted original folder after compression: {folder_path}")
            
        return os.path.abspath(zip_file_path)
        
    except Exception as e:
        # Clean up failed ZIP file if it exists
        if os.path.exists(zip_file_path):
            os.remove(zip_file_path)
        raise Exception(f"Compression failed: {str(e)}")
