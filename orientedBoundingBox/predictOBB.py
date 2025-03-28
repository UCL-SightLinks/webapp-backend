from ultralytics import YOLO
import numpy as np
from PIL import Image
from tqdm import tqdm
import os
import sys
import traceback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from georeference.georeference import georeferenceTIF, georefereceJGW, BNGtoLatLong
from utils.filterOutput import removeDuplicateBoxesRC, combineChunksToBaseName

def predictionJGW(imageAndDatas, predictionThreshold=0.25, saveLabeledImage=False, outputFolder="run/output", modelType="n", boundBoxChunkSize=1024, classificationChunkSize=256):
    """
    This function will take all of the segmented image and their georeferencing data from imageAndDatas, where the model then 
    processes the image and  creates a list of bounding boxes. It then takes each bounding box, georeferences it, and then 
    stores it in the dictionary imageDetectionsRowCol, where the key stores the basename, row, and column which this bounding 
    box came from. After looping through all of the items, it is then filtered to reduce duplications. This filter also 
    removes the row and column data, storing all of the bounding boxes from one image with the image name as the key.

    Args:
        imageAndDatas (list): A list of the input image name, segmented image, georeferencing data, row, and column.
        predictionThreshold (float): The confidence threshold for the bounding box model.
        saveLabeledImage (bool): If true, the images with bounding boxes will be saved.
        outputFolder (str): This directs where the model should save the output to.
        modelType (str): The type of model used.
        boundBoxChunkSize (int): The size of each side of the bounding box image.
        classificationChunkSize (int): The size of each side of the classification image.
    
    Returns:
        imageDetections (dict): A dictionary where the basename of an image is the key, and the key stores a list of boxes in latitude and longitude, and their respective confidence
    """
    # Get the absolute path to the models directory
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    modelPath = os.path.join(models_dir, f"yolo-{modelType}.pt")
    
    print(f"\nLoading YOLO model from: {modelPath}")
    if not os.path.exists(modelPath):
        print(f"Model file not found at: {modelPath}")
        print(f"Available model files: {os.listdir(models_dir)}")
        raise FileNotFoundError(f"YOLO model file not found at: {modelPath}")
        
    model = YOLO(modelPath)  # load an official model
    # Dictionary to store all detections and their confidence grouped by original image
    imageDetectionsRowCol = {}
    numOfSavedImages = 1
    # First, process all images and group detections
    with tqdm(total=(len(imageAndDatas)), desc="Creating Oriented Bounding Box") as pbar:
        for baseName, croppedImage, pixelSizeX, pixelSizeY, topLeftXGeo, topLeftYGeo, row, col in imageAndDatas:
            try:
                allPointsList = []
                allConfidenceList = []
                results = model(croppedImage, save=saveLabeledImage, conf=predictionThreshold, iou=0.01, 
                            project=outputFolder+"/labeledImages", name="run", exist_ok=True, verbose=False)
                if saveLabeledImage and os.path.exists(outputFolder+"/labeledImages/run/image0.jpg"):
                    os.rename(outputFolder+"/labeledImages/run/image0.jpg", outputFolder+f"/labeledImages/run/image{numOfSavedImages}.jpg")
                    numOfSavedImages += 1
                for result in results:
                    result = result.cpu()
                    for confidence in result.obb.conf:
                        allConfidenceList.append(confidence.item())
                    for boxes in result.obb.xyxyxyxy:
                        x1, y1 = boxes[0].tolist()
                        x2, y2 = boxes[1].tolist()
                        x3, y3 = boxes[2].tolist()
                        x4, y4 = boxes[3].tolist()
                        listOfPoints = georefereceJGW(x1,y1,x2,y2,x3,y3,x4,y4,pixelSizeX,pixelSizeY,topLeftXGeo,topLeftYGeo)
                        latLongList = BNGtoLatLong(listOfPoints)
                        allPointsList.append(latLongList)
                if allPointsList:
                    baseNameWithRowCol = f"{baseName}__r{row}__c{col}"
                    imageDetectionsRowCol[baseNameWithRowCol] = [allPointsList,allConfidenceList]
            except Exception as e:
                print(f"Error processing {croppedImage}: {e}")
                print(traceback.format_exc())
            pbar.update(1)
        
    removeDuplicateBoxesRC(imageDetectionsRowCol=imageDetectionsRowCol, boundBoxChunkSize=boundBoxChunkSize, classificationChunkSize=classificationChunkSize)
    imageDetections = combineChunksToBaseName(imageDetectionsRowCol=imageDetectionsRowCol)
    return imageDetections

# This version of predictionTIF has filtering
def predictionTIF(imageAndDatas, predictionThreshold=0.25, saveLabeledImage=False, outputFolder="run/output", modelType="n", boundBoxChunkSize=1024, classificationChunkSize=256):
    """
    This function will take all of the segmented image and their georeferencing data from imageAndDatas, where the model then 
    processes the image and  creates a list of bounding boxes. It then takes each bounding box, georeferences it, and then 
    stores it in the dictionary imageDetectionsRowCol, where the key stores the basename, row, and column which this bounding 
    box came from. After looping through all of the items, it is then filtered to reduce duplications. This filter also 
    removes the row and column data, storing all of the bounding boxes from one image with the image name as the key.

    Args:
        imageAndDatas (list): A list of the input image name, segmented tif image, row, and column.
        predictionThreshold (float): The confidence threshold for the bounding box model.
        saveLabeledImage (bool): If true, the images with bounding boxes will be saved.
        outputFolder (str): This directs where the model should save the output to.
        modelType (str): The type of model used.
        boundBoxChunkSize (int): The size of each side of the bounding box image.
        classificationChunkSize (int): The size of each side of the classification image.
    
    Returns:
        imageDetections (dict): A dictionary where the basename of an image is the key, and the key stores a list of boxes in latitude and longitude, and their respective confidence.
    """
    # Get the absolute path to the models directory
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    modelPath = os.path.join(models_dir, f"yolo-{modelType}.pt")
    
    print(f"\nStarting prediction with {len(imageAndDatas)} chunks")
    print(f"Model path: {modelPath}")
    print(f"Prediction threshold: {predictionThreshold}")
    
    if not os.path.exists(modelPath):
        print(f"Model file not found at: {modelPath}")
        print(f"Available model files: {os.listdir(models_dir)}")
        raise FileNotFoundError(f"YOLO model file not found at: {modelPath}")
        
    model = YOLO(modelPath)  # load an official model
    # Dictionary to store all detections and their confidence grouped by image, row, and column
    imageDetectionsRowCol = {}
    numOfSavedImages = 1
    
    # First, process all images and group detections
    with tqdm(total=(len(imageAndDatas)), desc="Creating Oriented Bounding Box") as pbar:
        for baseName, croppedImage, row, col in imageAndDatas:
            try:
                allPointsList = []
                allConfidenceList = []
                
                # Read the TIF data
                croppedImageArray = croppedImage.ReadAsArray()
                print(f"\nProcessing chunk from {baseName} at row {row}, col {col}")
                print(f"Array shape: {croppedImageArray.shape}")
                
                # Convert to RGB if needed
                if croppedImageArray.ndim == 3:
                    if croppedImageArray.shape[0] == 4:  # RGBA
                        croppedImageArray = croppedImageArray[:3]  # Take only RGB channels
                    croppedImageArray = np.moveaxis(croppedImageArray, 0, -1)  # Move channels to last dimension
                elif croppedImageArray.ndim == 2:  # Single band
                    croppedImageArray = np.stack([croppedImageArray] * 3, axis=-1)
                
                # Convert to PIL Image
                PILImage = Image.fromarray(croppedImageArray)
                print(f"PIL Image size: {PILImage.size}")
                
                # Run YOLO model
                results = model(PILImage, save=saveLabeledImage, conf=predictionThreshold, iou=0.9, 
                              project=outputFolder+"/labeledImages", name="run", exist_ok=True, verbose=False)
                
                if saveLabeledImage and os.path.exists(outputFolder+"/labeledImages/run/image0.jpg"):
                    os.rename(outputFolder+"/labeledImages/run/image0.jpg", outputFolder+f"/labeledImages/run/image{numOfSavedImages}.jpg")
                    numOfSavedImages += 1
                
                # Process results
                for result in results:
                    result = result.cpu()
                    for confidence in result.obb.conf:
                        allConfidenceList.append(confidence.item())
                    for boxes in result.obb.xyxyxyxy:
                        x1, y1 = boxes[0].tolist()
                        x2, y2 = boxes[1].tolist()
                        x3, y3 = boxes[2].tolist()
                        x4, y4 = boxes[3].tolist()
                        longLatList = georeferenceTIF(croppedImage, x1, y1, x2, y2, x3, y3, x4, y4)
                        allPointsList.append(longLatList)
                
                if allPointsList:
                    print(f"Found {len(allPointsList)} detections in this chunk")
                    baseNameWithRowCol = f"{baseName}__r{row}__c{col}"
                    imageDetectionsRowCol[baseNameWithRowCol] = [allPointsList, allConfidenceList]
                else:
                    print("No detections found in this chunk")
                    
            except Exception as e:
                print(f"Error processing {baseName}: {e}")
                print(traceback.format_exc())
            pbar.update(1)
    
    print(f"\nTotal chunks processed: {len(imageAndDatas)}")
    print(f"Chunks with detections: {len(imageDetectionsRowCol)}")
    
    removeDuplicateBoxesRC(imageDetectionsRowCol=imageDetectionsRowCol, boundBoxChunkSize=boundBoxChunkSize, classificationChunkSize=classificationChunkSize)
    imageDetections = combineChunksToBaseName(imageDetectionsRowCol=imageDetectionsRowCol)
    
    print(f"Final number of images with detections: {len(imageDetections)}")
    return imageDetections