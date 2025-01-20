from ultralytics import YOLO
from tqdm import tqdm
import os
import json
import sys
import re

# Load a model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from georeference.georeference import georeferecePoints
from georeference.georeference import BNGtoLatLong

def getOriginalImageName(filename):
    """Extract original image name from segmented filename"""
    # Remove row and column information (e.g., '_r1_c1')
    return re.sub(r'_r\d+_c\d+$', '', filename)

def saveTXTOutput(outputFolder, imageName, coordinates, confidences=None):
    """Save coordinates and optional confidence scores to a TXT file with one bounding box per line"""
    txtPath = os.path.join(outputFolder, f"{imageName}.txt")
    
    with open(txtPath, 'w') as file:
        for i, coordSet in enumerate(coordinates):
            # Format each point as "lon,lat" and join with spaces
            line = " ".join([f"{point[0]},{point[1]}" for point in coordSet])
            # Add confidence score if available
            if confidences is not None and i < len(confidences):
                line += f" {confidences[i]}"
            file.write(line + "\n")

def prediction(predictionThreshold=0.25, saveLabeledImage=False, outputType="0", outputFolder="run/output", modelType="n", progress_callback=None):
    # Convert outputType to int if it's a string
    outputType = int(outputType) if isinstance(outputType, str) else outputType
    
    modelPath = f"models/yolo-{modelType}.pt"
    model = YOLO(modelPath)  # load an official model
    
    # Dictionary to store all detections and their confidence grouped by original image
    imageDetections = {}
    processedImages = set()  # Use set to avoid duplicates
    
    # Get total number of images to process
    total_images = len([f for f in os.listdir(outputFolder) if f.endswith(('.jpg', '.jpeg', '.png'))])
    current_image = 0
    
    print("\n=== Processing Images ===")
    # First, process all images and group detections
    with tqdm(total=(total_images//2), desc="Creating Oriented Bounding Box") as pbar:
        for image in os.listdir(outputFolder):
            if not image.endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            current_image += 1
            if progress_callback:
                progress_callback(current_image, total_images)
                
            imagePath = os.path.join(outputFolder, image)
            try:
                # Get original image name
                originalImage = getOriginalImageName(os.path.splitext(image)[0])
                print(f"\nProcessing image: {originalImage}")
                processedImages.add(originalImage)  # Add to processed images set
                
                allPointsList = []
                allConfidenceList = []
                results = model(imagePath, save=saveLabeledImage, conf=predictionThreshold, iou=0.9, 
                              project=outputFolder+"/labeledImages", name="run", exist_ok=True, verbose=False)
                
                for result in results:
                    result = result.cpu()
                    for confidence in result.obb.conf:
                        allConfidenceList.append(confidence.item())
                    for boxes in result.obb.xyxyxyxy:
                        x1, y1 = boxes[0]
                        x2, y2 = boxes[1]
                        x3, y3 = boxes[2]
                        x4, y4 = boxes[3]
                        listOfPoints = georeferecePoints(x1,y1,x2,y2,x3,y3,x4,y4,imagePath)
                        longLatList = BNGtoLatLong(listOfPoints)
                        allPointsList.append(longLatList)
                
                # Initialize or update detections for this image
                if originalImage not in imageDetections:
                    imageDetections[originalImage] = [[],[]]
                if allPointsList:
                    imageDetections[originalImage][0].extend(allPointsList)
                    imageDetections[originalImage][1].extend(allConfidenceList)
                
                print(f"Found {len(allPointsList)} detections")
                print(f"Confidence scores: {allConfidenceList}")
                
                os.remove(imagePath)
                os.remove(imagePath.replace('jpg', 'jgw'))
                pbar.update(1)
            except Exception as e:
                print(f"Error processing {imagePath}: {e}")
    
    print("\n=== Saving Results ===")
    # Now save the grouped detections
    if outputType == 0:
        # Save as JSON
        jsonOutput = []
        # Include all processed images in output
        for originalImage in sorted(processedImages):  # Sort for consistent output
            coordAndConf = imageDetections.get(originalImage, [[],[]])
            entry = {
                "image": f"{originalImage}.jpg",
                "coordinates": coordAndConf[0],
                "confidence": coordAndConf[1]
            }
            jsonOutput.append(entry)
            print(f"\nImage: {originalImage}")
            print(f"Coordinates: {len(coordAndConf[0])} points")
            print(f"Confidence scores: {len(coordAndConf[1])} values")
        
        jsonPath = os.path.join(outputFolder, "output.json")
        print(f"\nWriting JSON output to: {jsonPath}")
        print("JSON content:")
        print(json.dumps(jsonOutput, indent=2))
        
        # Validate JSON content before writing
        if not jsonOutput:
            print("WARNING: No detections found in any processed images")
            jsonOutput = [{"image": f"{img}.jpg", "coordinates": [], "confidence": []} for img in processedImages]
            print("Creating empty entries for all processed images")
        
        try:
            with open(jsonPath, 'w') as file:
                json.dump(jsonOutput, file, indent=2)
            
            # Verify the file was written correctly
            if os.path.exists(jsonPath):
                file_size = os.path.getsize(jsonPath)
                print(f"JSON file size: {file_size} bytes")
                
                # Read back and validate content
                with open(jsonPath, 'r') as file:
                    content = file.read()
                    if not content:
                        print("ERROR: File is empty after writing")
                    else:
                        try:
                            parsed = json.loads(content)
                            print(f"Successfully verified JSON content with {len(parsed)} entries")
                            for entry in parsed:
                                print(f"Entry for {entry['image']}: {len(entry['coordinates'])} detections")
                        except json.JSONDecodeError as e:
                            print(f"ERROR: Invalid JSON content: {e}")
            else:
                print("ERROR: File was not created")
        except Exception as e:
            print(f"ERROR writing JSON file: {e}")
            raise
    else:
        # Save as TXT files
        for originalImage, coordAndConf in sorted(imageDetections.items()):  # Sort for consistent output
            saveTXTOutput(outputFolder, originalImage, coordAndConf[0], coordAndConf[1])
        print(f"\nTXT files saved to: {outputFolder}")
    
    print(f"\nProcessed {len(processedImages)} original images")
    return outputFolder