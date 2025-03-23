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
import shutil
sys.dont_write_bytecode = True

os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "YES"
os.environ["GDAL_MAX_IMAGE_PIXELS"] = "None"
Image.MAX_IMAGE_PIXELS = None

boundBoxChunkSize = 1024
classificationChunkSize = 256


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
    
    extractDir = None
    
    try:
        if inputType == "0":
            start_time = time.time()
            outputFolder = create_dir("run/output")
            extractDir = create_dir("run/extract")
            print(f"Extract directory created: {extractDir}")
            
            # Add a cancellation check point here before extraction
            if hasattr(sys, '_task_cancelled_callback') and callable(sys._task_cancelled_callback):
                sys._task_cancelled_callback()
                
            # Extract files if needed
            extractFiles(inputType, uploadDir, extractDir)
            
            # Add a cancellation check point after extraction
            if hasattr(sys, '_task_cancelled_callback') and callable(sys._task_cancelled_callback):
                sys._task_cancelled_callback()
                
            # Run segmentation and prediction
            croppedImagesAndData = boundBoxSegmentationJGW(classificationThreshold, extractDir, boundBoxChunkSize, classificationChunkSize)
            
            # Add a cancellation check point after segmentation
            if hasattr(sys, '_task_cancelled_callback') and callable(sys._task_cancelled_callback):
                sys._task_cancelled_callback()
                
            imageDetections = predictionJGW(imageAndDatas=croppedImagesAndData, predictionThreshold=predictionThreshold, saveLabeledImage=saveLabeledImage, outputFolder=outputFolder, modelType=yoloModelType)
            
            # Add a cancellation check point after prediction
            if hasattr(sys, '_task_cancelled_callback') and callable(sys._task_cancelled_callback):
                sys._task_cancelled_callback()
                
            # Save output and check if any detections were found
            has_detections = saveToOutput(outputType=outputType, outputFolder=outputFolder, imageDetections=imageDetections)
            
            if not has_detections:
                print("No detections found - not generating output files")
                # Return the output folder anyway so the no_detections.txt file can be accessed
                return outputFolder
                
            print(f"Output saved to {outputFolder} as {outputType}.")
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
            start_time = time.time()
            outputFolder = create_dir("run/output")
            extractDir = create_dir("run/extract")
            print(f"Extract directory created: {extractDir}")
            
            # Add a cancellation check point here before extraction
            if hasattr(sys, '_task_cancelled_callback') and callable(sys._task_cancelled_callback):
                sys._task_cancelled_callback()
                
            # Extract files if needed
            extractFiles(inputType, uploadDir, extractDir)
            
            # Add a cancellation check point after extraction
            if hasattr(sys, '_task_cancelled_callback') and callable(sys._task_cancelled_callback):
                sys._task_cancelled_callback()
                
            # Run segmentation and prediction
            croppedImagesAndData = boundBoxSegmentationTIF(classificationThreshold, extractDir, boundBoxChunkSize, classificationChunkSize)
            
            # Add a cancellation check point after segmentation
            if hasattr(sys, '_task_cancelled_callback') and callable(sys._task_cancelled_callback):
                sys._task_cancelled_callback()
                
            imageDetections = predictionTIF(imageAndDatas=croppedImagesAndData, predictionThreshold=predictionThreshold, saveLabeledImage=saveLabeledImage, outputFolder=outputFolder, modelType=yoloModelType)
            
            # Add a cancellation check point after prediction
            if hasattr(sys, '_task_cancelled_callback') and callable(sys._task_cancelled_callback):
                sys._task_cancelled_callback()
                
            # Save output and check if any detections were found
            has_detections = saveToOutput(outputType=outputType, outputFolder=outputFolder, imageDetections=imageDetections)
            
            if not has_detections:
                print("No detections found - not generating output files")
                # Return the output folder anyway so the no_detections.txt file can be accessed
                return outputFolder
                
            print(f"Output saved to {outputFolder} as {outputType}.")
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
        # Clean up extract directory in case of error
        if extractDir and os.path.exists(extractDir):
            try:
                shutil.rmtree(extractDir)
                print(f"Cleaned up extract directory after error: {extractDir}")
            except Exception as cleanup_err:
                print(f"Error cleaning up extract directory after error: {str(cleanup_err)}")
        raise  # Re-raise the original exception