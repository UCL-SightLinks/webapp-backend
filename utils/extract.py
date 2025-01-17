import os
import zipfile
import shutil
from tqdm import tqdm

def extract_files(input_type, upload_dir, extract_dir):
    """
    Extract or move files to the target directory based on input type
    Args:
        input_type (str): "0" for digimap data, "1" for custom data
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
        # Digimap data - only process zip files
        zip_files = [f for f in files if f.endswith('.zip')]
        if not zip_files:
            print("No zip files found")
            return
            
        # Process each zip file with outer progress bar
        for zip_file in tqdm(zip_files, desc="Processing zip files"):
            zip_path = os.path.join(upload_dir, zip_file)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Extract to a temporary directory
                temp_dir = 'temp_extract'
                zip_ref.extractall(temp_dir)
                
                # Get list of all files to process
                files_to_process = []
                for root, _, files in os.walk(temp_dir):
                    for filename in files:
                        if not filename.startswith('._') and filename.endswith(('.jpg', '.jgw')):
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
    else:
        # Custom data - check for zip files first
        files_to_process = []
        
        # Process files with progress bar
        with tqdm(total=len(files), desc="Processing files") as pbar:
            for file in files:
                if file.endswith('.zip'):
                    # Process zip file
                    zip_path = os.path.join(upload_dir, file)
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        temp_dir = 'temp_extract'
                        zip_ref.extractall(temp_dir)
                        
                        # Add extracted files to processing list
                        for root, _, extracted in os.walk(temp_dir):
                            for filename in extracted:
                                if not filename.startswith('._'):
                                    files_to_process.append((root, filename))
                        
                        # Process extracted files
                        for root, filename in files_to_process:
                            src_path = os.path.join(root, filename)
                            dst_path = os.path.join(extract_dir, filename)
                            shutil.copy2(src_path, dst_path)
                            if filename not in extracted_files:
                                extracted_files.add(filename)
                        
                        # Clean up temporary directory
                        shutil.rmtree(temp_dir)
                else:
                    # Only move jpg and jgw files if not a zip
                    if not file.startswith('._') and file.endswith(('.jpg', '.jgw')):
                        src_path = os.path.join(upload_dir, file)
                        dst_path = os.path.join(extract_dir, file)
                        shutil.copy2(src_path, dst_path)
                        if file not in extracted_files:
                            extracted_files.add(file)
                pbar.update(1)
    
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