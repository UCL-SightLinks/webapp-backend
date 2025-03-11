import os
import zipfile
import shutil
from tqdm import tqdm

def extract_files(input_type, upload_dir, extract_dir):
    """
    Extract or move files to the target directory based on input type
    Args:
        input_type (str): "0" for image data (jpg/png/jpeg + jgw), "1" for GeoTIFF data
        upload_dir (str): Directory where input files are located
        extract_dir (str): Directory to copy extracted files to
    """
    # Create target directory if it doesn't exist
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)
        
    # Check if upload directory exists
    if not os.path.exists(upload_dir):
        print(f"Upload directory {upload_dir} does not exist")
        return
    
    print(f"Processing files from {upload_dir}")
    extracted_files = set()  # Keep track of processed files
    
    # Get list of all files in upload directory
    files = os.listdir(upload_dir)
    
    if input_type == "0":
        # Type 0: Process jpg/png/jpeg + jgw files or zip containing these
        valid_extensions = ('.jpg', '.jpeg', '.png', '.jgw')
        
        # Process zip files first
        zip_files = [f for f in files if f.endswith('.zip')]
        for zip_file in tqdm(zip_files, desc="Processing zip files"):
            zip_path = os.path.join(upload_dir, zip_file)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Extract to a temporary directory
                temp_dir = 'temp_extract'
                zip_ref.extractall(temp_dir)
                
                # Get list of all files to process
                files_to_process = []
                for root, _, temp_files in os.walk(temp_dir):
                    for filename in temp_files:
                        if not filename.startswith('._') and filename.lower().endswith(valid_extensions):
                            files_to_process.append((root, filename))
                
                # Process files with inner progress bar
                for root, filename in tqdm(files_to_process, desc=f"Extracting from {zip_file}", leave=False):
                    src_path = os.path.join(root, filename)
                    dst_path = os.path.join(extract_dir, filename)
                    shutil.copy2(src_path, dst_path)
                    if filename not in extracted_files:
                        extracted_files.add(filename)
                
                # Clean up temporary directory
                shutil.rmtree(temp_dir)
        
        # Now process individual image/jgw files
        individual_files = [f for f in files if f.lower().endswith(valid_extensions)]
        for file in tqdm(individual_files, desc="Processing individual files"):
            if not file.startswith('._'):
                src_path = os.path.join(upload_dir, file)
                dst_path = os.path.join(extract_dir, file)
                shutil.copy2(src_path, dst_path)
                if file not in extracted_files:
                    extracted_files.add(file)
                    
    else:  # input_type == "1"
        # Type 1: Process GeoTIFF files or zip containing them
        valid_extensions = ('.tif', '.tiff')
        
        # Process zip files first
        zip_files = [f for f in files if f.endswith('.zip')]
        for zip_file in tqdm(zip_files, desc="Processing zip files"):
            zip_path = os.path.join(upload_dir, zip_file)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Extract to a temporary directory
                temp_dir = 'temp_extract'
                zip_ref.extractall(temp_dir)
                
                # Get list of all files to process
                files_to_process = []
                for root, _, temp_files in os.walk(temp_dir):
                    for filename in temp_files:
                        if not filename.startswith('._') and filename.lower().endswith(valid_extensions):
                            files_to_process.append((root, filename))
                
                # Process files with inner progress bar
                for root, filename in tqdm(files_to_process, desc=f"Extracting from {zip_file}", leave=False):
                    src_path = os.path.join(root, filename)
                    dst_path = os.path.join(extract_dir, filename)
                    shutil.copy2(src_path, dst_path)
                    if filename not in extracted_files:
                        extracted_files.add(filename)
                
                # Clean up temporary directory
                shutil.rmtree(temp_dir)
        
        # Now process individual GeoTIFF files
        individual_files = [f for f in files if f.lower().endswith(valid_extensions)]
        for file in tqdm(individual_files, desc="Processing individual files"):
            if not file.startswith('._'):
                src_path = os.path.join(upload_dir, file)
                dst_path = os.path.join(extract_dir, file)
                shutil.copy2(src_path, dst_path)
                if file not in extracted_files:
                    extracted_files.add(file)
    
    print("\nProcessed files:")
    for filename in sorted(extracted_files):
        print(f"- {filename}")
    print(f"\nTotal files processed: {len(extracted_files)}")
    print(f"Files saved to: {extract_dir}")

"""
example usage:
if __name__ == "__main__":
    extract_files("1", "input")
"""