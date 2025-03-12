import math
from itertools import combinations
import re
from shapely.geometry import Polygon
import yaml
from pathlib import Path
from torchvision.ops import nms
from torch import is_tensor, tensor
from tqdm import tqdm
import os

def combineChunksToBaseName(imageDetectionsRowCol):
    """
    Combine detection results from multiple chunks into a single dictionary based on base name.
    
    Args:
        imageDetectionsRowCol (dict): Dictionary with chunked detection results where the key is 
                                      a base name with row and column information, and the value 
                                      contains detection points and confidence.
    
    Returns:
        dict: A dictionary with base names as keys and corresponding detection points and 
              confidence as values across all chunks.
    """
    if not imageDetectionsRowCol:
        print("Warning: Empty imageDetectionsRowCol dictionary")
        return {}

    imageDetections = {}
    for nameWithRowCol in imageDetectionsRowCol:
        try:
            if not isinstance(nameWithRowCol, str):
                print(f"Warning: Invalid key type in imageDetectionsRowCol: {type(nameWithRowCol)}")
                continue

            if nameWithRowCol not in imageDetectionsRowCol:
                print(f"Warning: Key {nameWithRowCol} not found in imageDetectionsRowCol")
                continue

            data = imageDetectionsRowCol[nameWithRowCol]
            if not isinstance(data, (list, tuple)) or len(data) != 2:
                print(f"Warning: Invalid data format for key {nameWithRowCol}")
                continue

            baseName, _, _ = extractBaseNameAndCoords(nameWithRowCol)
            imageDetections.setdefault(baseName, [[], []])
            imageDetections[baseName][0].extend(data[0])
            imageDetections[baseName][1].extend(data[1])
        except Exception as e:
            print(f"Error processing chunk {nameWithRowCol}: {str(e)}")
            try:
                # Use the full name as the base name if extraction fails
                imageDetections.setdefault(nameWithRowCol, [[], []])
                imageDetections[nameWithRowCol][0].extend(imageDetectionsRowCol[nameWithRowCol][0])
                imageDetections[nameWithRowCol][1].extend(imageDetectionsRowCol[nameWithRowCol][1])
            except Exception as inner_e:
                print(f"Error in fallback processing for chunk {nameWithRowCol}: {str(inner_e)}")
                continue
    
    if not imageDetections:
        print("Warning: No valid detections were processed")
    
    return imageDetections

def checkBoxIntersection(box1, box2, threshold=0.6):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes and check if their intersection 
    exceeds a threshold.
    
    Args:
        box1 (list): The first bounding box defined by four corners.
        box2 (list): The second bounding box defined by four corners.
        threshold (float): The IoU threshold to determine if the boxes are considered as overlapping.

    Returns:
        bool: True if the IoU exceeds the threshold, otherwise False.
    """
    if not box1 or not box2:
        print("Warning: Empty box coordinates provided")
        return False

    try:
        poly1 = Polygon(box1)
        poly2 = Polygon(box2)
        
        if not poly1.is_valid or not poly2.is_valid:
            print("Warning: Invalid polygon geometry")
            return False
            
        intersectionArea = poly1.intersection(poly2).area
        unionArea = poly1.union(poly2).area
        
        if unionArea <= 0:
            return False
            
        iou = intersectionArea / unionArea
        
        # Avoid division by zero
        minArea = min(poly1.area, poly2.area)
        if minArea <= 0:
            return False
            
        scaledIou = iou * (max(poly1.area, poly2.area) / minArea)
        return scaledIou > threshold
    except Exception as e:
        print(f"Error in checkBoxIntersection: {str(e)}")
        return False

def extractBaseNameAndCoords(baseNameWithRowCol):
    """
    Extract the base name, row, and column from a string formatted as '{baseName}__r{row}__c{col}'.
    
    Args:
        baseNameWithRowCol (str): A string in the format '{baseName}__r{row}__c{col}'.
    
    Returns:
        tuple: A tuple containing the base name (str), row (int), and column (int).
    """
    if not baseNameWithRowCol:
        print("Warning: Empty baseNameWithRowCol provided")
        return "unknown", 0, 0

    if not isinstance(baseNameWithRowCol, str):
        print(f"Warning: Invalid type for baseNameWithRowCol: {type(baseNameWithRowCol)}")
        return str(baseNameWithRowCol), 0, 0

    try:
        match = re.match(r"(.+?)__r(-?\d+)__c(-?\d+)", baseNameWithRowCol)
        if match:
            baseName = match.group(1)
            row = int(match.group(2))
            col = int(match.group(3))
            return baseName, row, col
        else:
            print(f"Warning: Input string '{baseNameWithRowCol}' is not in the expected format. Using as base name with default coordinates.")
            # If it's a path, try to extract just the filename
            if os.path.sep in baseNameWithRowCol:
                baseNameWithRowCol = os.path.basename(baseNameWithRowCol)
            return baseNameWithRowCol, 0, 0
    except Exception as e:
        print(f"Error parsing baseNameWithRowCol '{baseNameWithRowCol}': {str(e)}")
        return str(baseNameWithRowCol), 0, 0

def removeDuplicateBoxesRC(imageDetectionsRowCol):
    """
    Remove duplicate bounding boxes that overlap with neighboring chunks in an 11x11 grid.
    
    Args:
        imageDetectionsRowCol (dict): Dictionary with chunked detection results.
    """
    if not imageDetectionsRowCol:
        print("Warning: Empty imageDetectionsRowCol dictionary")
        return

    with tqdm(total=len(imageDetectionsRowCol), desc="Filtering crosswalks") as pbar:
        for currentKeyToFilter in list(imageDetectionsRowCol.keys()):  # Create a list to avoid modification during iteration
            try:
                if currentKeyToFilter not in imageDetectionsRowCol:
                    print(f"Warning: Key {currentKeyToFilter} no longer in dictionary")
                    continue

                data = imageDetectionsRowCol[currentKeyToFilter]
                if not isinstance(data, (list, tuple)) or len(data) != 2:
                    print(f"Warning: Invalid data format for key {currentKeyToFilter}")
                    continue

                allPointsList, allConfidenceList = data
                if not isinstance(allPointsList, list) or not isinstance(allConfidenceList, list):
                    print(f"Warning: Invalid points or confidence format for key {currentKeyToFilter}")
                    continue

                toRemove = set()
                baseName, row, col = extractBaseNameAndCoords(currentKeyToFilter)
                toRemoveNeighboringMap = {}

                # Process neighboring chunks
                for dRow in range(-5, 6):
                    for dCol in range(-5, 6):
                        if dRow == 0 and dCol == 0:
                            continue

                        try:
                            newRow, newCol = row + dRow, col + dCol
                            neighboringChunk = f"{baseName}__r{newRow}__c{newCol}"
                            if neighboringChunk not in imageDetectionsRowCol:
                                continue

                            neighboringData = imageDetectionsRowCol[neighboringChunk]
                            if not isinstance(neighboringData, (list, tuple)) or len(neighboringData) != 2:
                                continue

                            neighboringBoxes, neighboringConf = neighboringData
                            toRemoveNeighboring = set()

                            # Compare boxes
                            for i, boxA in enumerate(allPointsList):
                                if i in toRemove:
                                    continue
                                for j, boxB in enumerate(neighboringBoxes):
                                    if j in toRemoveNeighboring:
                                        continue

                                    if checkBoxIntersection(boxA, boxB, threshold=0.7):
                                        if allConfidenceList[i] <= neighboringConf[j]:
                                            toRemove.add(i)
                                            break
                                        else:
                                            toRemoveNeighboring.add(j)
                                            break

                            if toRemoveNeighboring:
                                toRemoveNeighboringMap[neighboringChunk] = toRemoveNeighboringMap.get(neighboringChunk, set()).union(toRemoveNeighboring)

                        except Exception as neighbor_error:
                            print(f"Error processing neighbor for {currentKeyToFilter}: {str(neighbor_error)}")
                            continue

                # Update dictionaries
                try:
                    for chunk, indices in toRemoveNeighboringMap.items():
                        if chunk in imageDetectionsRowCol:
                            chunkData = imageDetectionsRowCol[chunk]
                            if isinstance(chunkData, (list, tuple)) and len(chunkData) == 2:
                                imageDetectionsRowCol[chunk][0] = [box for i, box in enumerate(chunkData[0]) if i not in indices]
                                imageDetectionsRowCol[chunk][1] = [conf for i, conf in enumerate(chunkData[1]) if i not in indices]

                    imageDetectionsRowCol[currentKeyToFilter][0] = [box for i, box in enumerate(allPointsList) if i not in toRemove]
                    imageDetectionsRowCol[currentKeyToFilter][1] = [conf for i, conf in enumerate(allConfidenceList) if i not in toRemove]
                except Exception as update_error:
                    print(f"Error updating dictionaries for {currentKeyToFilter}: {str(update_error)}")

            except Exception as e:
                print(f"Error filtering duplicates for {currentKeyToFilter}: {str(e)}")
                continue

            pbar.update(1)
