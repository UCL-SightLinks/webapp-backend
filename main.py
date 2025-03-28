from imageSegmentation.boundBoxSegmentation import boundBoxSegmentationJGW, boundBoxSegmentationTIF
from orientedBoundingBox.predictOBB import predictionJGW, predictionTIF
from utils.extract import extractFiles
from utils.saveToOutput import saveToOutput
from datetime import datetime
from PIL import Image
import os
import time
import sys
import shutil
import traceback
sys.dont_write_bytecode = True

os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "YES"
os.environ["GDAL_MAX_IMAGE_PIXELS"] = "None"
Image.MAX_IMAGE_PIXELS = None

boundBoxChunkSize = 1024
classificationChunkSize = 256

# Get application root directory
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

def check_for_cancellation():
    """Utility function to check for task cancellation"""
    if hasattr(sys, '_task_cancelled_callback') and callable(sys._task_cancelled_callback):
        sys._task_cancelled_callback()

def create_dir(run_dir):
    """Create and return timestamped output directory"""
    # Make sure the run_dir is an absolute path
    if not os.path.isabs(run_dir):
        run_dir = os.path.join(APP_ROOT, run_dir)
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    i = 0
    while os.path.exists(run_dir+"/"+timestamp):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")+"_"+str(i)
        i += 1
    output_dir = os.path.join(run_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created: {output_dir}")
    return output_dir

def execute(uploadDir = "input", inputType = "0", classificationThreshold = 0.35, predictionThreshold = 0.5, saveLabeledImage = False, outputType = "0", yoloModelType = "m"):
    # Convert parameters to their proper types
    inputType = str(inputType)
    classificationThreshold = float(classificationThreshold)
    predictionThreshold = float(predictionThreshold)
    
    # Convert saveLabeledImage to boolean if it's a string
    if isinstance(saveLabeledImage, str):
        saveLabeledImage = saveLabeledImage.lower() == 'true'
    else:
        saveLabeledImage = bool(saveLabeledImage)
        
    outputType = str(outputType)
    yoloModelType = str(yoloModelType)
    
    # Make sure uploadDir is an absolute path
    if not os.path.isabs(uploadDir):
        uploadDir = os.path.join(APP_ROOT, uploadDir)
    
    print(f"\n=== Starting Main Processing ===")
    print(f"Parameters:")
    print(f"- Upload directory: {uploadDir}")
    print(f"- Input type: {inputType}")
    print(f"- Classification threshold: {classificationThreshold}")
    print(f"- Prediction threshold: {predictionThreshold}")
    print(f"- Save labeled image: {saveLabeledImage}")
    print(f"- Output type: {outputType}")
    print(f"- YOLO model type: {yoloModelType}")
    print(f"- Upload directory exists: {os.path.exists(uploadDir)}")
    
    # Check if multiple ZIP files are present
    zip_files = [f for f in os.listdir(uploadDir) if f.lower().endswith('.zip')]
    print(f"\nFound {len(zip_files)} ZIP files in upload directory")
    
    # If multiple ZIP files, process each separately
    if len(zip_files) > 1:
        print("\nProcessing multiple ZIP files...")
        all_output_folders = []
        for zip_file in zip_files:
            print(f"\nProcessing ZIP file: {zip_file}")
            # Create a temporary directory for this ZIP
            temp_dir = os.path.join(uploadDir, f"temp_{os.path.splitext(zip_file)[0]}")
            os.makedirs(temp_dir, exist_ok=True)
            
            # Copy the ZIP file to the temporary directory
            shutil.copy2(os.path.join(uploadDir, zip_file), os.path.join(temp_dir, zip_file))
            
            # Process this single ZIP file
            try:
                output_folder = process_single_input(temp_dir, inputType, classificationThreshold, 
                                                    predictionThreshold, saveLabeledImage, 
                                                    outputType, yoloModelType)
                all_output_folders.append(output_folder)
                print(f"Successfully processed ZIP file: {zip_file}")
            except Exception as e:
                print(f"Error processing ZIP file {zip_file}: {str(e)}")
                print(traceback.format_exc())
            finally:
                # Clean up temporary directory
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                    print(f"Cleaned up temporary directory: {temp_dir}")
        
        # Create a merged output folder with results from all ZIPs
        merged_output = create_dir("run/output")
        print(f"\nCreating merged output folder: {merged_output}")
        
        # Copy all results to the merged folder
        has_detections = False
        for folder in all_output_folders:
            if os.path.exists(folder):
                # Check if this folder has detections
                if not os.path.exists(os.path.join(folder, "no_detections.txt")):
                    has_detections = True
                
                # Copy all files from this folder to the merged folder
                for item in os.listdir(folder):
                    src_path = os.path.join(folder, item)
                    dst_path = os.path.join(merged_output, item)
                    
                    if os.path.isfile(src_path):
                        shutil.copy2(src_path, dst_path)
                    elif os.path.isdir(src_path):
                        shutil.copytree(src_path, dst_path)
                print(f"Copied results from {folder} to merged output")
        
        # If no detections were found in any ZIP, create a no_detections.txt file
        if not has_detections:
            with open(os.path.join(merged_output, "no_detections.txt"), 'w') as f:
                f.write("No detections found in any of the uploaded files")
            print("No detections found in any ZIP file")
        
        print(f"\nMerged output saved to: {merged_output}")
        return merged_output
    else:
        # Process normally for single ZIP or non-ZIP files
        print("\nProcessing single input...")
        return process_single_input(uploadDir, inputType, classificationThreshold, 
                                  predictionThreshold, saveLabeledImage, 
                                  outputType, yoloModelType)

def process_single_input(uploadDir, inputType, classificationThreshold, predictionThreshold, saveLabeledImage, outputType, yoloModelType):
    """Process a single input directory (either containing a single ZIP or regular files)"""
    extractDir = None
    
    try:
        if inputType == "0":
            print("\n=== Processing JGW Input ===")
            start_time = time.time()
            outputFolder = create_dir("run/output")
            extractDir = create_dir("run/extract")
            print(f"Created directories:")
            print(f"- Output folder: {outputFolder}")
            print(f"- Extract directory: {extractDir}")
            
            # Add a cancellation check point here before extraction
            check_for_cancellation()
                
            # Extract files if needed
            print("\nExtracting files...")
            extractFiles(inputType, uploadDir, extractDir)
            print("Files extracted successfully")
            
            # Add a cancellation check point after extraction
            check_for_cancellation()
                
            # Run segmentation and prediction
            print("\nRunning segmentation...")
            croppedImagesAndData = boundBoxSegmentationJGW(classificationThreshold, extractDir, boundBoxChunkSize, classificationChunkSize)
            print(f"Segmentation complete. Found {len(croppedImagesAndData)} images to process")
            
            # Add a cancellation check point after segmentation
            check_for_cancellation()
                
            print("\nRunning prediction...")
            imageDetections = predictionJGW(imageAndDatas=croppedImagesAndData, predictionThreshold=predictionThreshold, saveLabeledImage=saveLabeledImage, outputFolder=outputFolder, modelType=yoloModelType)
            print("Prediction complete")
            
            # Add a cancellation check point after prediction
            check_for_cancellation()
                
            # Save output and check if any detections were found
            print("\nSaving output...")
            has_detections = saveToOutput(outputType=outputType, outputFolder=outputFolder, imageDetections=imageDetections)
            
            if not has_detections:
                print("No detections found - not generating output files")
                # Return the output folder anyway so the no_detections.txt file can be accessed
                return outputFolder
                
            print(f"Output saved to {outputFolder} as {outputType}")
            print(f"Total time taken: {time.time() - start_time:.2f} seconds")
            
            # Clean up extract directory
            if extractDir and os.path.exists(extractDir):
                try:
                    shutil.rmtree(extractDir)
                    print(f"Cleaned up extract directory: {extractDir}")
                except Exception as e:
                    print(f"Error cleaning up extract directory: {str(e)}")
                    
            return outputFolder

        elif inputType == "1":
            print("\n=== Processing TIF Input ===")
            start_time = time.time()
            outputFolder = create_dir("run/output")
            extractDir = create_dir("run/extract")
            print(f"Created directories:")
            print(f"- Output folder: {outputFolder}")
            print(f"- Extract directory: {extractDir}")
            
            # Add a cancellation check point here before extraction
            check_for_cancellation()
                
            # Extract files if needed
            print("\nExtracting files...")
            extractFiles(inputType, uploadDir, extractDir)
            print("Files extracted successfully")
            
            # Add a cancellation check point after extraction
            check_for_cancellation()
                
            # Run segmentation and prediction
            print("\nRunning segmentation...")
            croppedImagesAndData = boundBoxSegmentationTIF(classificationThreshold, extractDir, boundBoxChunkSize, classificationChunkSize)
            print(f"Segmentation complete. Found {len(croppedImagesAndData)} images to process")
            
            # Add a cancellation check point after segmentation
            check_for_cancellation()
                
            print("\nRunning prediction...")
            imageDetections = predictionTIF(imageAndDatas=croppedImagesAndData, predictionThreshold=predictionThreshold, saveLabeledImage=saveLabeledImage, outputFolder=outputFolder, modelType=yoloModelType)
            print("Prediction complete")
            
            # Add a cancellation check point after prediction
            check_for_cancellation()
                
            # Save output and check if any detections were found
            print("\nSaving output...")
            has_detections = saveToOutput(outputType=outputType, outputFolder=outputFolder, imageDetections=imageDetections)
            
            if not has_detections:
                print("No detections found - not generating output files")
                # Return the output folder anyway so the no_detections.txt file can be accessed
                return outputFolder
                
            print(f"Output saved to {outputFolder} as {outputType}")
            print(f"Total time taken: {time.time() - start_time:.2f} seconds")
            
            # Clean up extract directory
            if extractDir and os.path.exists(extractDir):
                try:
                    shutil.rmtree(extractDir)
                    print(f"Cleaned up extract directory: {extractDir}")
                except Exception as e:
                    print(f"Error cleaning up extract directory: {str(e)}")
                    
            return outputFolder
    except Exception as e:
        print(f"\n=== Error in process_single_input ===")
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        
        # Clean up extract directory in case of error
        if extractDir and os.path.exists(extractDir):
            try:
                shutil.rmtree(extractDir)
                print(f"Cleaned up extract directory after error: {extractDir}")
            except Exception as cleanup_err:
                print(f"Error cleaning up extract directory after error: {str(cleanup_err)}")
        raise  # Re-raise the original exception