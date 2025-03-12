from imageSegmentation.boundBoxSegmentation import boundBoxSegmentationJGW, boundBoxSegmentationTIF
from orientedBoundingBox.predictOBB import predictionJGW, predictionTIF
from utils.extract import extractFiles
from utils.saveToOutput import saveToOutput
from datetime import datetime
from PIL import Image
from osgeo import gdal
import os
import time
import sys
sys.dont_write_bytecode = True

os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "YES"
os.environ["GDAL_MAX_IMAGE_PIXELS"] = "None"
Image.MAX_IMAGE_PIXELS = None

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



def execute(uploadDir = "input", inputType = "0", classificationThreshold = 0.35, predictionThreshold = 0.5, saveLabeledImage = False, outputType = "0", yoloModelType = "n", progress_callback = None, cancellation_check = None):
    """
    Execute the image processing pipeline with progress tracking and cancellation support.
    
    Args:
        uploadDir: Directory containing uploaded files
        inputType: Type of input (0 for JGW, 1 for TIF)
        classificationThreshold: Threshold for classification
        predictionThreshold: Threshold for prediction
        saveLabeledImage: Whether to save labeled images
        outputType: Type of output format
        yoloModelType: YOLO model type to use (n, s, m)
        progress_callback: Function to call with progress updates (stage, percentage)
        cancellation_check: Function to call to check if task should be cancelled
    
    Returns:
        Path to the output folder
    """
    extractDir = None
    outputFolder = None
    
    try:
        # Convert parameters to proper types
        inputType = str(inputType)
        classificationThreshold = float(classificationThreshold)
        predictionThreshold = float(predictionThreshold)
        saveLabeledImage = bool(saveLabeledImage)
        outputType = str(outputType)
        
        # Initialize processing
        if progress_callback:
            progress_callback("Initializing processing environment", 2)
        
        # Check for cancellation
        if cancellation_check and cancellation_check():
            raise Exception("Task cancelled by user")
        
        start_time = time.time()
        outputFolder = create_dir("run/output")
        extractDir = create_dir("run/extract")
        
        if progress_callback:
            progress_callback("Starting file extraction", 5)
        
        # Check for cancellation
        if cancellation_check and cancellation_check():
            raise Exception("Task cancelled by user")
        
        # Extract files if needed
        try:
            extractFiles(inputType, uploadDir, extractDir)
        except Exception as extract_error:
            error_msg = f"File extraction failed: {str(extract_error)}"
            print(error_msg)
            if progress_callback:
                progress_callback(error_msg, 15)
            raise Exception(error_msg)
        
        # Check that files exist in the extract directory
        extracted_files = os.listdir(extractDir)
        if not extracted_files:
            error_msg = "No files were extracted from the input"
            print(error_msg)
            if progress_callback:
                progress_callback(error_msg, 15)
            raise Exception(error_msg)
            
        # For JGW files, check for matching pairs
        if inputType == "0":
            image_files = [f for f in extracted_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            jgw_files = [f for f in extracted_files if f.lower().endswith('.jgw')]
            
            if not image_files:
                error_msg = "No image files found in the extract directory"
                print(error_msg)
                if progress_callback:
                    progress_callback(error_msg, 15)
                raise Exception(error_msg)
                
            if not jgw_files:
                error_msg = "No JGW files found in the extract directory"
                print(error_msg)
                if progress_callback:
                    progress_callback(error_msg, 15)
                raise Exception(error_msg)
                
            # Check for matching pairs
            image_bases = [os.path.splitext(f)[0] for f in image_files]
            jgw_bases = [os.path.splitext(f)[0] for f in jgw_files]
            matching_pairs = set(image_bases).intersection(set(jgw_bases))
            
            if not matching_pairs:
                error_msg = "No matching image-JGW pairs found"
                print(error_msg)
                if progress_callback:
                    progress_callback(error_msg, 15)
                raise Exception(error_msg)
                
            print(f"Found {len(matching_pairs)} matching image-JGW pairs")
        
        # For TIF files, check for valid TIF files
        elif inputType == "1":
            tif_files = [f for f in extracted_files if f.lower().endswith('.tif')]
            
            if not tif_files:
                error_msg = "No TIF files found in the extract directory"
                print(error_msg)
                if progress_callback:
                    progress_callback(error_msg, 15)
                raise Exception(error_msg)
                
            print(f"Found {len(tif_files)} TIF files")
        
        if progress_callback:
            progress_callback("File extraction completed", 15)
            progress_callback("Initializing image segmentation", 20)
        
        # Check for cancellation
        if cancellation_check and cancellation_check():
            raise Exception("Task cancelled by user")
        
        # Process based on input type
        if inputType == "0":  # JGW format
            # Define segmentation progress callback for JGW
            def segmentation_progress_callback_jgw(current, total):
                if progress_callback:
                    # Scale progress from 20 to 40 based on segmentation progress
                    progress = 20 + (current / max(1, total) * 20)
                    progress_callback(f"Segmenting image {current}/{total}", int(progress))
                    
                if cancellation_check and cancellation_check():
                    raise Exception("Task cancelled by user")
            
            # Run segmentation with progress tracking
            try:
                croppedImagesAndData = boundBoxSegmentationJGW(
                    classificationThreshold, 
                    extractDir, 
                    progress_callback=segmentation_progress_callback_jgw
                )
                
                if not croppedImagesAndData or len(croppedImagesAndData) == 0:
                    error_msg = "No valid image segments were produced during segmentation"
                    print(error_msg)
                    if progress_callback:
                        progress_callback(error_msg, 40)
                    raise Exception(error_msg)
                    
            except Exception as e:
                error_msg = f"Error in image segmentation: {str(e)}"
                print(error_msg)
                if progress_callback:
                    progress_callback(error_msg, 40)
                raise
            
            if progress_callback:
                progress_callback("Image segmentation completed", 40)
                progress_callback(f"Loading YOLO model type: {yoloModelType}", 45)
            
            # Check for cancellation
            if cancellation_check and cancellation_check():
                raise Exception("Task cancelled by user")
            
            # Define prediction progress callback for JGW
            def prediction_progress_callback_jgw(current_image, total_images):
                if progress_callback:
                    # Scale progress from 50 to 80 based on prediction processing
                    progress = 50 + (current_image / max(1, total_images) * 30)
                    progress_callback(f"Processing image {current_image}/{total_images} with YOLO", int(progress))
                    
                if cancellation_check and cancellation_check():
                    raise Exception("Task cancelled by user")
            
            if progress_callback:
                progress_callback("Starting image processing with YOLO model", 50)
            
            # Run prediction with progress tracking
            try:
                imageDetections = predictionJGW(
                    croppedImagesAndData, 
                    predictionThreshold, 
                    saveLabeledImage, 
                    outputFolder, 
                    yoloModelType,
                    progress_callback=prediction_progress_callback_jgw
                )
                
                if imageDetections is None:
                    error_msg = "YOLO processing failed to produce any detections"
                    print(error_msg)
                    if progress_callback:
                        progress_callback(error_msg, 80)
                    raise Exception(error_msg)
                    
            except Exception as e:
                error_msg = f"Error in YOLO processing: {str(e)}"
                print(error_msg)
                if progress_callback:
                    progress_callback(error_msg, 80)
                raise
            
        elif inputType == "1":  # TIF format
            # Define segmentation progress callback for TIF
            def segmentation_progress_callback_tif(current, total):
                if progress_callback:
                    # Scale progress from 20 to 40 based on segmentation progress
                    progress = 20 + (current / max(1, total) * 20)
                    progress_callback(f"Segmenting TIF image {current}/{total}", int(progress))
                    
                if cancellation_check and cancellation_check():
                    raise Exception("Task cancelled by user")
            
            # Run segmentation with progress tracking
            try:
                croppedImagesAndData = boundBoxSegmentationTIF(
                    classificationThreshold, 
                    extractDir,
                    progress_callback=segmentation_progress_callback_tif
                )
                
                if not croppedImagesAndData or len(croppedImagesAndData) == 0:
                    error_msg = "No valid TIF image segments were produced during segmentation"
                    print(error_msg)
                    if progress_callback:
                        progress_callback(error_msg, 40)
                    raise Exception(error_msg)
                    
            except Exception as e:
                error_msg = f"Error in TIF image segmentation: {str(e)}"
                print(error_msg)
                if progress_callback:
                    progress_callback(error_msg, 40)
                raise
            
            if progress_callback:
                progress_callback("TIF image segmentation completed", 40)
                progress_callback(f"Loading YOLO model type: {yoloModelType}", 45)
            
            # Check for cancellation
            if cancellation_check and cancellation_check():
                raise Exception("Task cancelled by user")
            
            # Define prediction progress callback for TIF
            def prediction_progress_callback_tif(current_image, total_images):
                if progress_callback:
                    # Scale progress from 50 to 80 based on prediction processing
                    progress = 50 + (current_image / max(1, total_images) * 30)
                    progress_callback(f"Processing TIF image {current_image}/{total_images} with YOLO", int(progress))
                    
                if cancellation_check and cancellation_check():
                    raise Exception("Task cancelled by user")
            
            if progress_callback:
                progress_callback("Starting TIF image processing with YOLO model", 50)
            
            # Run prediction with progress tracking
            try:
                imageDetections = predictionTIF(
                    imageAndDatas=croppedImagesAndData, 
                    predictionThreshold=predictionThreshold, 
                    saveLabeledImage=saveLabeledImage, 
                    outputFolder=outputFolder, 
                    modelType=yoloModelType,
                    progress_callback=prediction_progress_callback_tif
                )
                
                if imageDetections is None:
                    error_msg = "YOLO processing failed to produce any TIF detections"
                    print(error_msg)
                    if progress_callback:
                        progress_callback(error_msg, 80)
                    raise Exception(error_msg)
                    
            except Exception as e:
                error_msg = f"Error in TIF YOLO processing: {str(e)}"
                print(error_msg)
                if progress_callback:
                    progress_callback(error_msg, 80)
                raise
        else:
            raise ValueError(f"Invalid input type: {inputType}")
        
        if progress_callback:
            progress_callback("YOLO processing completed", 80)
            progress_callback("Generating output files", 85)
        
        # Check for cancellation
        if cancellation_check and cancellation_check():
            raise Exception("Task cancelled by user")
        
        if progress_callback:
            progress_callback("Saving output files", 90)
        
        # Save output files
        try:
            saveToOutput(outputType=outputType, outputFolder=outputFolder, imageDetections=imageDetections)
        except Exception as e:
            error_msg = f"Error saving output: {str(e)}"
            print(error_msg)
            if progress_callback:
                progress_callback(error_msg, 90)
            # Don't raise here, try to continue with cleanup
            print("Attempting to continue despite saveToOutput error...")
        
        if progress_callback:
            progress_callback("Finalizing results", 95)
        
        print(f"Output saved to {outputFolder} as {outputType}.")
        print(f"Total time taken: {time.time() - start_time:.2f} seconds")

        if progress_callback:
            progress_callback("Processing completed successfully", 100)
        
        return outputFolder
        
    except Exception as e:
        print(f"Error in execute: {str(e)}")
        import traceback
        traceback.print_exc()
        if progress_callback:
            progress_callback(f"Error occurred: {str(e)}", 100)
        raise
    finally:
        # Clean up extract directory if it exists and an error occurred
        if extractDir and os.path.exists(extractDir):
            try:
                import shutil
                # Only clean up if we failed to create or save to the output folder
                if not outputFolder or not os.path.exists(outputFolder) or not os.listdir(outputFolder):
                    shutil.rmtree(extractDir)
                    print(f"Directory cleaned up: {extractDir}")
            except Exception as cleanup_error:
                print(f"Error cleaning up directory {extractDir}: {str(cleanup_error)}")
                # Don't raise cleanup errors as they're not critical
                pass