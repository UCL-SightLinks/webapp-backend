from imageSegmentation.boundBoxSegmentation import boundBoxSegmentation
from orientedBoundingBox.predictOBB import prediction
from utils.extract import extract_files
from datetime import datetime
import os
import time
import sys
import shutil
sys.dont_write_bytecode = True

def create_dir(run_dir):
    """Create and return timestamped output directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    i = 0
    while os.path.exists(run_dir+"/"+timestamp):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")+"_"+str(i)
        i += 1
    output_dir = os.path.join(run_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created: {output_dir}")
    return output_dir

def clean_up(dir):
    """Clean up the output directory"""
    shutil.rmtree(dir)
    print(f"Directory cleaned up: {dir}")

def execute(uploadDir = "input", inputType = "0", classificationThreshold = 0.35, predictionThreshold = 0.5, saveLabeledImage = False, outputType = "0", yoloModelType = "n", progress_callback = None, cancellation_check = None):
    start_time = time.time()
    
    if progress_callback:
        progress_callback("Initializing processing environment", 2)
    
    # Check for cancellation
    if cancellation_check and cancellation_check():
        raise Exception("Task cancelled by user")
    
    outputFolder = create_dir("run/output")
    extractDir = create_dir("run/extract")
    
    # Convert parameters to proper types
    outputType = int(outputType)  # Convert to integer
    inputType = int(inputType)
    classificationThreshold = float(classificationThreshold)
    predictionThreshold = float(predictionThreshold)
    saveLabeledImage = str(saveLabeledImage).lower() == 'true'
    
    try:
        # Extract files if needed
        if progress_callback:
            progress_callback("Starting file extraction", 5)
        
        # Check for cancellation
        if cancellation_check and cancellation_check():
            raise Exception("Task cancelled by user")
            
        extract_files(inputType, uploadDir, extractDir)
        
        if progress_callback:
            progress_callback("File extraction completed", 15)
            progress_callback("Initializing image segmentation", 20)
        
        # Check for cancellation
        if cancellation_check and cancellation_check():
            raise Exception("Task cancelled by user")
            
        # Run segmentation
        boundBoxSegmentation(classificationThreshold, outputFolder, extractDir)
        
        if progress_callback:
            progress_callback("Image segmentation completed", 40)
            progress_callback("Loading YOLO model", 45)
            progress_callback(f"Initializing YOLO model type: {yoloModelType}", 48)
        
        # Check for cancellation
        if cancellation_check and cancellation_check():
            raise Exception("Task cancelled by user")
            
        # Run prediction with progress tracking
        if progress_callback:
            progress_callback("Starting image processing with YOLO model", 50)
            
        def obb_progress_callback(current_image, total_images):
            if progress_callback:
                # Scale progress from 50 to 80 based on OBB processing
                progress = 50 + (current_image / total_images * 30)
                progress_callback(f"Processing image {current_image}/{total_images}", int(progress))
                
            if cancellation_check and cancellation_check():
                raise Exception("Task cancelled by user")
                
        output_path = prediction(predictionThreshold, saveLabeledImage, outputType, outputFolder, yoloModelType, obb_progress_callback)
        
        if progress_callback:
            progress_callback("YOLO processing completed", 80)
            progress_callback("Generating output files", 85)
            progress_callback("Preparing final results", 90)
        
        print(f"Output saved to {outputFolder} as {outputType}.")
        print(f"Total time taken: {time.time() - start_time:.2f} seconds")
        
        if progress_callback:
            progress_callback("Cleaning up temporary files", 95)
        clean_up(extractDir)
        
        if progress_callback:
            progress_callback("Processing completed successfully", 98)
        
        return output_path
    except Exception as e:
        print(f"Error in execute: {str(e)}")
        if progress_callback:
            progress_callback(f"Error occurred: {str(e)}", 100)
        clean_up(extractDir)
        raise e
