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
    if inputType == "0":
        start_time = time.time()
        outputFolder = create_dir("run/output")
        extractDir = create_dir("run/extract")
        # Extract files if needed
        extractFiles(inputType, uploadDir, extractDir)
        # Run segmentation and prediction
        croppedImagesAndData = boundBoxSegmentationJGW(classificationThreshold, extractDir, boundBoxChunkSize, classificationChunkSize)
        imageDetections = predictionJGW(imageAndDatas=croppedImagesAndData, predictionThreshold=predictionThreshold, saveLabeledImage=saveLabeledImage, outputFolder=outputFolder, modelType=yoloModelType)
        
        saveToOutput(outputType=outputType, outputFolder=outputFolder, imageDetections=imageDetections)
        print(f"Output saved to {outputFolder} as {outputType}.")
        print(f"Total time taken: {time.time() - start_time:.2f} seconds")

    elif inputType == "1":
        start_time = time.time()
        outputFolder = create_dir("run/output")
        extractDir = create_dir("run/extract")
        # Extract files if needed
        extractFiles(inputType, uploadDir, extractDir)
        # Run segmentation and prediction
        croppedImagesAndData = boundBoxSegmentationTIF(classificationThreshold, extractDir, boundBoxChunkSize, classificationChunkSize)
        imageDetections = predictionTIF(imageAndDatas=croppedImagesAndData, predictionThreshold=predictionThreshold, saveLabeledImage=saveLabeledImage, outputFolder=outputFolder, modelType=yoloModelType)
        
        saveToOutput(outputType=outputType, outputFolder=outputFolder, imageDetections=imageDetections)
        print(f"Output saved to {outputFolder} as {outputType}.")
        print(f"Total time taken: {time.time() - start_time:.2f} seconds")