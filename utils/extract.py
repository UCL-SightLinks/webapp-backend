import os
import zipfile
import shutil
import sys
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# from imageSegmentation.tifResize import getPixelCount, tileResize

def extractFiles(inputType, uploadDir, extractDir):
    """
    Extract or move files to the target directory based on input type
    Args:
        inputType (str): "0" for jpg and jgw data, "1" for geotiff data
        uploadDir (str): Directory where input files are located
        extractDir (str): Directory to copy extracted files to
    
    Returns:
        bool: True if files were successfully extracted, False otherwise
    
    Raises:
        Exception: If no supported files are found after extraction
    """
    # Create target directory if it doesn't exist
    if not os.path.exists(extractDir):
        os.makedirs(extractDir)
        
    # Check if upload directory exists
    if not os.path.exists(uploadDir):
        raise Exception(f"Upload directory {uploadDir} does not exist")
    
    print(f"Processing files from {uploadDir}")
    extractedFiles = set()  # Keep track of processed files
    
    # Get list of all files in upload directory
    files = os.listdir(uploadDir)
    
    if len(files) == 0:
        raise Exception(f"No files found in upload directory {uploadDir}")
    
    if inputType == "0":
        # Custom data - check for zip files first
        filesToProcess = []
        zipFound = False
        
        # Process files with progress bar
        with tqdm(total=len(files), desc="Processing files") as pbar:
            for file in files:
                if file.endswith('.zip'):
                    zipFound = True
                    # Process zip file
                    zipPath = os.path.join(uploadDir, file)
                    
                    # Check if zip file is empty or corrupted
                    try:
                        with zipfile.ZipFile(zipPath, 'r') as zipRef:
                            fileList = zipRef.namelist()
                            if not fileList:
                                print(f"Warning: Zip file {file} is empty")
                                pbar.update(1)
                                continue
                                
                            # Create a temporary directory for extraction
                            tempDir = os.path.join(extractDir, 'temp_extract')
                            os.makedirs(tempDir, exist_ok=True)
                            
                            # Extract files
                            print(f"Extracting {len(fileList)} files from {file}")
                            zipRef.extractall(tempDir)
                            
                            # Add extracted files to processing list
                            validFileFound = False
                            for root, _, extracted in os.walk(tempDir):
                                for filename in extracted:
                                    if not filename.startswith('._'):
                                        if filename.endswith(('.jpg', '.jpeg', '.png', '.jgw')):
                                            validFileFound = True
                                        filesToProcess.append((root, filename))
                            
                            # Process extracted files
                            for root, filename in filesToProcess:
                                srcPath = os.path.join(root, filename)
                                dstPath = os.path.join(extractDir, filename)
                                shutil.copy2(srcPath, dstPath)
                                if filename not in extractedFiles:
                                    extractedFiles.add(filename)
                            
                            # Clean up temporary directory
                            shutil.rmtree(tempDir)
                            
                            if not validFileFound:
                                print(f"Warning: No supported image files found in zip file {file}")
                    except zipfile.BadZipFile:
                        print(f"Error: {file} is not a valid zip file")
                        pbar.update(1)
                        continue
                    except Exception as e:
                        print(f"Error processing zip file {file}: {str(e)}")
                        pbar.update(1)
                        continue
                else:
                    # Only move jpg and jgw files if not a zip
                    if not file.startswith('._') and file.endswith(('.jpg', '.jpeg', '.png', '.jgw')):
                        srcPath = os.path.join(uploadDir, file)
                        dstPath = os.path.join(extractDir, file)
                        shutil.copy2(srcPath, dstPath)
                        if file not in extractedFiles:
                            extractedFiles.add(file)
                pbar.update(1)

    elif inputType == "1":
        extractedFiles = set()
        zipFound = False

        # Process files with progress bar
        with tqdm(total=len(files), desc="Processing files") as pbar:
            for file in files:
                if file.endswith('.zip'):
                    zipFound = True
                    # Process zip file
                    zipPath = os.path.join(uploadDir, file)
                    
                    # Check if zip file is empty or corrupted
                    try:
                        with zipfile.ZipFile(zipPath, 'r') as zipRef:
                            fileList = zipRef.namelist()
                            if not fileList:
                                print(f"Warning: Zip file {file} is empty")
                                pbar.update(1)
                                continue
                                
                            # Create a temporary directory for extraction
                            tempDir = os.path.join(extractDir, 'temp_extract')
                            os.makedirs(tempDir, exist_ok=True)
                            
                            # Extract files
                            print(f"Extracting {len(fileList)} files from {file}")
                            zipRef.extractall(tempDir)

                            # Add extracted files to processing list
                            filesToProcess = []
                            validFileFound = False
                            for root, _, extracted in os.walk(tempDir):
                                for filename in extracted:
                                    if not filename.startswith('._'):
                                        if filename.endswith('.tif'):
                                            validFileFound = True
                                        filesToProcess.append((root, filename))

                            # Process extracted files
                            for root, filename in filesToProcess:
                                srcPath = os.path.join(root, filename)
                                dstPath = os.path.join(extractDir, filename)
                                shutil.copy2(srcPath, dstPath)
                                if filename not in extractedFiles:
                                    extractedFiles.add(filename)

                            # Clean up temporary directory
                            shutil.rmtree(tempDir)
                            
                            if not validFileFound:
                                print(f"Warning: No supported TIFF files found in zip file {file}")
                    except zipfile.BadZipFile:
                        print(f"Error: {file} is not a valid zip file")
                        pbar.update(1)
                        continue
                    except Exception as e:
                        print(f"Error processing zip file {file}: {str(e)}")
                        pbar.update(1)
                        continue
                elif file.endswith('.tif'):
                    srcPath = os.path.join(uploadDir, file)
                    dstPath = os.path.join(extractDir, file)
                    shutil.copy2(srcPath, dstPath)
                    if file not in extractedFiles:
                        extractedFiles.add(file)
                
                # Check pixel count and apply tileResize if necessary        
                pbar.update(1)
    
    print("\nProcessed files:")
    for filename in sorted(extractedFiles):
        print(f"- {filename}")
    print(f"\nTotal files processed: {len(extractedFiles)}")
    print(f"Files saved to: {extractDir}")
    
    # Check if we have valid image files based on input type
    validExtensions = ['.jpg', '.jpeg', '.png'] if inputType == "0" else ['.tif']
    validImageFiles = [f for f in extractedFiles if any(f.lower().endswith(ext) for ext in validExtensions)]
    
    if not validImageFiles:
        zipInfo = "A zip file was found but contained no valid image files." if zipFound else "No zip files were found."
        foundFiles = ", ".join(extractedFiles) if extractedFiles else "none"
        raise Exception(f"No supported image files found in the extract directory. {zipInfo} Files found: {foundFiles}")
    
    print(f"Found {len(validImageFiles)} valid image files")
    return True


"""
example usage:
if __name__ == "__main__":
    extract_files("1", "input")
"""