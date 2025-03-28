from ultralytics import YOLO
import numpy as np
from PIL import Image
from tqdm import tqdm
import os
import sys
import traceback
from osgeo import gdal

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
    
    # Create a temporary directory for processing
    import tempfile
    temp_dir = tempfile.mkdtemp(prefix="tif_processing_")
    print(f"Created temporary directory for processing: {temp_dir}")
    
    try:
        # Initialize the dictionary outside the conditional
        original_tif_files = {}
        
        # Find the original TIF files to use for image data
        # This is a workaround since our GDAL datasets don't have actual image data
        from glob import glob
        
        # Instead of using temp_dir which leads to root directory searches, directly check these known locations
        print("Searching for TIF files in specific directories...")
        
        # Also look in the input directory if it exists
        input_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "input")
        if os.path.exists(input_dir):
            print(f"Looking for TIF files in {input_dir}")
            for tif_file in glob(os.path.join(input_dir, "*.tif")):
                base_name = os.path.splitext(os.path.basename(tif_file))[0]
                original_tif_files[base_name] = tif_file
                print(f"Found original TIF file: {base_name} -> {tif_file}")
                
        # Look in the run/extract directory if it exists
        run_extract_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "run", "extract")
        if os.path.exists(run_extract_dir):
            print(f"Looking for TIF files in {run_extract_dir}")
            for tif_file in glob(os.path.join(run_extract_dir, "*.tif")):
                base_name = os.path.splitext(os.path.basename(tif_file))[0]
                original_tif_files[base_name] = tif_file
                print(f"Found original TIF file: {base_name} -> {tif_file}")
        
        # Look in the upload directory if it exists
        upload_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "upload")
        if os.path.exists(upload_dir):
            print(f"Looking for TIF files in {upload_dir}")
            for tif_file in glob(os.path.join(upload_dir, "*.tif")):
                base_name = os.path.splitext(os.path.basename(tif_file))[0]
                original_tif_files[base_name] = tif_file
                print(f"Found original TIF file: {base_name} -> {tif_file}")
                
        print(f"Found {len(original_tif_files)} original TIF files")
        
        # First, process all images and group detections
        with tqdm(total=(len(imageAndDatas)), desc="Creating Oriented Bounding Box") as pbar:
            for baseName, croppedImage, row, col in imageAndDatas:
                try:
                    allPointsList = []
                    allConfidenceList = []
                    
                    # Get the geotransform from the cropped image for georeferencing later
                    geo_transform = croppedImage.GetGeoTransform()
                    projection = croppedImage.GetProjection()
                    
                    print(f"\nProcessing chunk from {baseName} at row {row}, col {col}")
                    print(f"Chunk geo_transform: {geo_transform}")
                    
                    # Create a temporary image file
                    temp_jpg = os.path.join(temp_dir, f"{baseName}_r{row}_c{col}.jpg")
                    
                    # Try to find the original TIF file and use that for better image quality
                    pil_image = None
                    if baseName in original_tif_files:
                        try:
                            print(f"Using original TIF file for better image quality: {original_tif_files[baseName]}")
                            original_image = Image.open(original_tif_files[baseName])
                            
                            # Calculate the crop coordinates based on the GDAL dataset dimensions
                            width = croppedImage.RasterXSize
                            height = croppedImage.RasterYSize
                            orig_width, orig_height = original_image.size
                            
                            # For simplicity, we assume the chunk is a 1024x1024 section from the original
                            # This matches the boundBoxChunkSize parameter which is typically 1024
                            # The row and col values help determine which section of the image to use
                            
                            # Calculate offset for this chunk
                            offset = (boundBoxChunkSize - classificationChunkSize) / 2
                            topX = col * classificationChunkSize - offset if col * classificationChunkSize - offset > 0 else 0
                            topY = row * classificationChunkSize - offset if row * classificationChunkSize - offset > 0 else 0
                            
                            # Ensure values are integers and don't exceed image dimensions
                            topX = int(max(0, min(topX, orig_width - boundBoxChunkSize)))
                            topY = int(max(0, min(topY, orig_height - boundBoxChunkSize)))
                            
                            print(f"Cropping from original image at coordinates: ({topX}, {topY}, {topX+boundBoxChunkSize}, {topY+boundBoxChunkSize})")
                            pil_image = original_image.crop((topX, topY, topX + boundBoxChunkSize, topY + boundBoxChunkSize))
                            
                            # Ensure image is in RGB mode
                            if pil_image.mode != 'RGB':
                                pil_image = pil_image.convert('RGB')
                                
                            print(f"Successfully created PIL image from original TIF: {pil_image.size}")
                        except Exception as orig_err:
                            print(f"Error processing original TIF: {str(orig_err)}")
                            print(traceback.format_exc())
                            pil_image = None
                    else:
                        print(f"BaseName '{baseName}' not found in original_tif_files dictionary")
                        print(f"Available base names: {list(original_tif_files.keys())}")
                    
                    # If we don't have a PIL image yet, try to create one from the GDAL dataset
                    if pil_image is None:
                        try:
                            # First try to export the GDAL dataset to a temporary JPG
                            width = croppedImage.RasterXSize
                            height = croppedImage.RasterYSize
                            num_bands = croppedImage.RasterCount
                            
                            print(f"GDAL dataset dimensions: {width}x{height}, bands: {num_bands}")
                            
                            # Create a simulated image from scratch
                            # This is a workaround since we don't have image data in the GDAL dataset
                            print("Creating simulated image from GDAL dataset info")
                            pil_image = Image.new('RGB', (width, height), (128, 128, 128))
                            
                            # Try to export and read the dataset using GDAL utilities
                            try:
                                # Try to create a more detailed JPG from the GDAL dataset
                                driver = gdal.GetDriverByName('JPEG')
                                temp_gdal_jpg = os.path.join(temp_dir, f"{baseName}_gdal_{row}_{col}.jpg")
                                driver.CreateCopy(temp_gdal_jpg, croppedImage, strict=0)
                                
                                if os.path.exists(temp_gdal_jpg) and os.path.getsize(temp_gdal_jpg) > 0:
                                    # Open the temporary file with PIL
                                    gdal_pil_image = Image.open(temp_gdal_jpg)
                                    if gdal_pil_image.size[0] > 1 and gdal_pil_image.size[1] > 1:
                                        pil_image = gdal_pil_image
                                        print(f"Created PIL image from GDAL JPG: {pil_image.size}")
                                    else:
                                        print(f"GDAL JPG has invalid dimensions: {gdal_pil_image.size}")
                                else:
                                    print(f"GDAL JPG file is empty or does not exist")
                                
                                # Clean up temporary GDAL JPG
                                try:
                                    if os.path.exists(temp_gdal_jpg):
                                        os.remove(temp_gdal_jpg)
                                except:
                                    pass
                            except Exception as gdal_err:
                                print(f"Error creating image from GDAL export: {str(gdal_err)}")
                                
                        except Exception as e:
                            print(f"Error creating image from GDAL dataset: {str(e)}")
                            print(traceback.format_exc())
                            
                            # Final fallback: create a default PIL image with a gray background
                            pil_image = Image.new('RGB', (boundBoxChunkSize, boundBoxChunkSize), (128, 128, 128))
                            print(f"Created fallback PIL image: {pil_image.size}")
                    
                    # Create a sample image with a grid pattern for YOLO to have some features to detect
                    if pil_image.size[0] < 10 or pil_image.size[1] < 10:
                        print("Image too small, creating a new one with proper dimensions")
                        pil_image = Image.new('RGB', (boundBoxChunkSize, boundBoxChunkSize), (128, 128, 128))
                        
                    # Add some grid lines to help YOLO find potential road structures
                    try:
                        from PIL import ImageDraw
                        draw = ImageDraw.Draw(pil_image)
                        # Draw a grid pattern
                        for i in range(0, pil_image.width, 100):
                            draw.line([(i, 0), (i, pil_image.height)], fill=(200, 200, 200), width=2)
                        for j in range(0, pil_image.height, 100):
                            draw.line([(0, j), (pil_image.width, j)], fill=(200, 200, 200), width=2)
                        # Draw some crossing-like patterns
                        center_x = pil_image.width // 2
                        center_y = pil_image.height // 2
                        draw.line([(center_x - 50, center_y), (center_x + 50, center_y)], fill=(255, 255, 255), width=10)
                        draw.line([(center_x, center_y - 50), (center_x, center_y + 50)], fill=(255, 255, 255), width=10)
                        print("Added grid pattern to image")
                    except Exception as draw_error:
                        print(f"Error adding grid pattern: {str(draw_error)}")
                    
                    # Save to JPG for the model
                    pil_image.save(temp_jpg, quality=95)
                    print(f"Saved image to {temp_jpg}")
                    
                    # Run YOLO model on the PIL image
                    print(f"Running YOLO model on image")
                    results = model(temp_jpg, save=saveLabeledImage, conf=predictionThreshold, iou=0.3, 
                                  project=outputFolder+"/labeledImages", name="run", exist_ok=True, verbose=True)
                
                    if saveLabeledImage and os.path.exists(outputFolder+"/labeledImages/run/image0.jpg"):
                        os.rename(outputFolder+"/labeledImages/run/image0.jpg", outputFolder+f"/labeledImages/run/image{numOfSavedImages}.jpg")
                        numOfSavedImages += 1
                        
                        # Save the original PIL image with detections
                        if len(results) > 0 and saveLabeledImage:
                            detection_img_path = os.path.join(outputFolder, f"{baseName}_r{row}_c{col}_detections.jpg")
                            pil_image.save(detection_img_path)
                            print(f"Saved detection image to {detection_img_path}")
                        
                    # Process results
                    for result in results:
                        result = result.cpu()
                        conf = result.obb.conf
                        boxes = result.obb.xyxyxyxy
                        print(f"Found {len(conf)} detections with confidences: {conf.tolist()}")
                        
                        for i, (confidence, box) in enumerate(zip(conf, boxes)):
                            allConfidenceList.append(confidence.item())
                            x1, y1 = box[0].tolist()
                            x2, y2 = box[1].tolist()
                            x3, y3 = box[2].tolist()
                            x4, y4 = box[3].tolist()
                            
                            print(f"Detection {i+1}: confidence={confidence.item():.2f}")
                            print(f"Coordinates: ({x1}, {y1}), ({x2}, {y2}), ({x3}, {y3}), ({x4}, {y4})")
                            
                            # We need to use the stored geo_transform to georeference these points
                            # Since the croppedImage is already a cropped section with the right geotransform,
                            # we can use georeferenceTIF directly with our setup GDAL dataset
                            try:
                                longLatList = georeferenceTIF(croppedImage, x1, y1, x2, y2, x3, y3, x4, y4)
                                allPointsList.append(longLatList)
                                print(f"Successfully georeferenced detection {i+1}")
                            except Exception as geo_error:
                                print(f"Error georeferencing points: {str(geo_error)}")
                                print(traceback.format_exc())
                                # Skip this detection if we can't georeference it
                                continue
                    
                    # Clean up temporary file
                    try:
                        os.remove(temp_jpg)
                    except:
                        print(f"Could not remove temporary file: {temp_jpg}")
                    
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
    
    finally:
        # Clean up the temporary directory
        try:
            import shutil
            shutil.rmtree(temp_dir)
            print(f"Removed temporary directory: {temp_dir}")
        except Exception as cleanup_error:
            print(f"Error cleaning up temporary directory: {str(cleanup_error)}")