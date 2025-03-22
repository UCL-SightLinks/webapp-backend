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
    """
    # Create target directory if it doesn't exist
    if not os.path.exists(extractDir):
        os.makedirs(extractDir)
        
    # Check if upload directory exists
    if not os.path.exists(uploadDir):
        print(f"Upload directory {uploadDir} does not exist")
        return
    
    print(f"Processing files from {uploadDir}")
    extractedFiles = set()  # Keep track of processed files
    
    # Get list of all files in upload directory
    files = os.listdir(uploadDir)
    
    if inputType == "0":
        # Custom data - check for zip files first
        filesToProcess = []
        
        # Process files with progress bar
        with tqdm(total=len(files), desc="Processing files") as pbar:
            for file in files:
                if file.endswith('.zip'):
                    # Process zip file
                    zipPath = os.path.join(uploadDir, file)
                    with zipfile.ZipFile(zipPath, 'r') as zipRef:
                        tempDir = 'temp_extract'
                        zipRef.extractall(tempDir)
                        
                        # Add extracted files to processing list
                        for root, _, extracted in os.walk(tempDir):
                            for filename in extracted:
                                if not filename.startswith('._'):
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

        # Process files with progress bar
        with tqdm(total=len(files), desc="Processing files") as pbar:
            for file in files:
                if file.endswith('.zip'):
                    # Process zip file
                    zipPath = os.path.join(uploadDir, file)
                    with zipfile.ZipFile(zipPath, 'r') as zipRef:
                        tempDir = 'temp_extract'
                        zipRef.extractall(tempDir)

                        # Add extracted files to processing list
                        filesToProcess = []
                        for root, _, extracted in os.walk(tempDir):
                            for filename in extracted:
                                if not filename.startswith('._'):
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


"""
example usage:
if __name__ == "__main__":
    extract_files("1", "input")
"""