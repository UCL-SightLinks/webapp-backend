import os
import zipfile

def compress_folder_to_zip(folder_path, zip_file_name = "results.zip"):
    """
    Compress a folder and its contents into a ZIP file.

    Args:
        folder_path (str): Path to the folder to compress.
        zip_file_name (str): Name of the resulting ZIP file (include .zip extension).

    Returns:
        str: Path to the created ZIP file.
    """
    try:
        print(f"Compressing {folder_path} to {zip_file_name}")
        # Ensure the folder exists
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"The folder {folder_path} does not exist.")

        # Create the ZIP file
        with zipfile.ZipFile(zip_file_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    # Add file to the ZIP archive, preserving the folder structure
                    arcname = os.path.relpath(file_path, start=folder_path)
                    zipf.write(file_path, arcname)

        print(f"ZIP file created: {os.path.abspath(zip_file_name)}")
        return os.path.abspath(zip_file_name)
    except Exception as e:
        return f"An error occurred: {e}"
        
