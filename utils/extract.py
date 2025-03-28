import os
import zipfile
import shutil
import sys
from tqdm import tqdm
from osgeo import gdal
import numpy as np
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.api.logger_handler import LoggerHandler
# from imageSegmentation.tifResize import getPixelCount, tileResize

logger_handler = LoggerHandler()

def extractFiles(inputType, uploadDir, extractDir):
    """
    Extract or move files to the target directory based on input type
    Args:
        inputType (str): "0" for jpg and jgw data, "1" for geotiff data
        uploadDir (str): Directory where input files are located
        extractDir (str): Directory to copy extracted files to
    """
    # Create target directory if it doesn't exist
    if not os.path.exists(extractDir):
        os.makedirs(extractDir)
        
    # Check if upload directory exists
    if not os.path.exists(uploadDir):
        logger_handler.log_error(f"Upload directory {uploadDir} does not exist")
        return
    
    logger_handler.log_system(f"Processing files from {uploadDir}")
    extractedFiles = set()  # Keep track of processed files
    
    # Get list of all files in upload directory
    files = os.listdir(uploadDir)
    file_extensions = [os.path.splitext(file)[1].lower() for file in files]
    logger_handler.log_system(f"Received files: {', '.join(files)}")
    logger_handler.log_system(f"File extensions: {', '.join(file_extensions)}")
    logger_handler.log_system(f"Input type: {inputType}")
    
    # Process files with progress bar
    with tqdm(total=len(files), desc="Processing files") as pbar:
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            logger_handler.log_system(f"Processing file: {file} with extension {ext}")
            
            # Process ZIP files
            if file.endswith('.zip'):
                zipPath = os.path.join(uploadDir, file)
                logger_handler.log_system(f"Processing ZIP file: {file}")
                
                try:
                    with zipfile.ZipFile(zipPath, 'r') as zipRef:
                        # Create separate temp directory for each zip file
                        tempDir = f'temp_extract_{os.path.basename(zipPath)}'
                        if os.path.exists(tempDir):
                            shutil.rmtree(tempDir)
                        os.makedirs(tempDir)
                        
                        # Extract files
                        zipRef.extractall(tempDir)
                        
                        # Add extracted files to processing list
                        filesToProcess = []
                        for root, _, extracted in os.walk(tempDir):
                            for filename in extracted:
                                if not filename.startswith('._'):
                                    filesToProcess.append((root, filename))
                        
                        # Process extracted files based on input type
                        for root, filename in filesToProcess:
                            if (inputType == "0" and filename.endswith(('.jpg', '.jpeg', '.png', '.jgw'))) or \
                               (inputType == "1" and filename.endswith(('.tif', '.tiff'))) or \
                               (filename.endswith(('.jpg', '.jpeg', '.png', '.jgw', '.tif', '.tiff'))):
                                
                                srcPath = os.path.join(root, filename)
                                dstPath = os.path.join(extractDir, filename)
                                shutil.copy2(srcPath, dstPath)
                                if filename not in extractedFiles:
                                    extractedFiles.add(filename)
                                    logger_handler.log_system(f"Extracted file from ZIP: {filename}")
                        
                        # Clean up temporary directory
                        shutil.rmtree(tempDir)
                except Exception as e:
                    logger_handler.log_error(f"Error processing ZIP file {file}: {str(e)}")
            
            # Process individual files based on input type
            elif (inputType == "0" and file.endswith(('.jpg', '.jpeg', '.png', '.jgw'))) or \
                 (inputType == "1" and file.endswith(('.tif', '.tiff'))):
                
                srcPath = os.path.join(uploadDir, file)
                dstPath = os.path.join(extractDir, file)
                file_size = os.path.getsize(srcPath) if os.path.exists(srcPath) else 0
                
                logger_handler.log_system(f"Processing individual file: {file}, size: {file_size} bytes")
                
                try:
                    # For TIF files, verify and process with GDAL
                    if file.lower().endswith(('.tif', '.tiff')):
                        # Open with GDAL to verify
                        dataset = gdal.Open(srcPath, gdal.GA_ReadOnly)
                        if dataset is None:
                            raise Exception(f"Failed to open TIF file: {srcPath}")
                            
                        # Log TIF file details
                        logger_handler.log_system(f"TIF file details:")
                        logger_handler.log_system(f"- Width: {dataset.RasterXSize}")
                        logger_handler.log_system(f"- Height: {dataset.RasterYSize}")
                        logger_handler.log_system(f"- Bands: {dataset.RasterCount}")
                        logger_handler.log_system(f"- CRS: {dataset.GetProjection()}")
                        
                        # Verify each band has data
                        for b in range(dataset.RasterCount):
                            band = dataset.GetRasterBand(b + 1)
                            stats = band.GetStatistics(True, True)
                            if stats[0] == 0 and stats[1] == 0:  # If min and max are both 0
                                raise Exception(f"Band {b+1} contains only zeros")
                        
                        dataset = None  # Close the dataset
                    
                    # Copy the file
                    shutil.copy2(srcPath, dstPath)
                    
                    # Verify the copied file
                    if os.path.exists(dstPath):
                        copied_size = os.path.getsize(dstPath)
                        logger_handler.log_system(f"File copied successfully. Size: {copied_size} bytes")
                        if copied_size != file_size:
                            raise Exception(f"File size mismatch after copy: original={file_size}, copied={copied_size}")
                    else:
                        raise Exception(f"File was not copied successfully: {dstPath}")
                        
                    if file not in extractedFiles:
                        extractedFiles.add(file)
                        logger_handler.log_system(f"Successfully processed file: {file}")
                except Exception as e:
                    logger_handler.log_error(f"Error processing file {file}: {str(e)}")
            else:
                # Skip files with unsupported extensions
                logger_handler.log_system(f"Skipping file: {file} - unsupported for input_type {inputType}")
            
            pbar.update(1)
    
    # Summary of processed files
    logger_handler.log_system(f"\nProcessed files summary:")
    for filename in sorted(extractedFiles):
        logger_handler.log_system(f"- {filename}")
    logger_handler.log_system(f"\nTotal files processed: {len(extractedFiles)}")
    logger_handler.log_system(f"Files saved to: {extractDir}")


"""
example usage:
if __name__ == "__main__":
    extract_files("1", "input")
"""