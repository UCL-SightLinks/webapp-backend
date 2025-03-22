import math
from itertools import combinations
import re
from shapely.geometry import Polygon
import yaml
from pathlib import Path
from torchvision.ops import nms
from torch import is_tensor, tensor
from tqdm import tqdm

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
    imageDetections = {}
    for nameWithRowCol in imageDetectionsRowCol:
        baseName, _, _ = extractBaseNameAndCoords(nameWithRowCol)
        imageDetections.setdefault(baseName, [[], []])
        imageDetections[baseName][0].extend(imageDetectionsRowCol[nameWithRowCol][0])
        imageDetections[baseName][1].extend(imageDetectionsRowCol[nameWithRowCol][1])
    
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
    poly1 = Polygon(box1)
    poly2 = Polygon(box2)
    try:
        intersectionArea = poly1.intersection(poly2).area
        unionArea = poly1.union(poly2).area
        iou = intersectionArea / unionArea if unionArea > 0 else 0

        scaledIou = iou * (max(poly1.area, poly2.area) / min(poly1.area, poly2.area))
        return scaledIou > threshold
    except:
        return False  # Return False if the intersection computation fails
    

    
def extractBaseNameAndCoords(baseNameWithRowCol):
    """
    Extract the base name, row, and column from a string formatted as '{baseName}__r{row}__c{col}'.
    
    Args:
        baseNameWithRowCol (str): A string in the format '{baseName}__r{row}__c{col}'.
    
    Returns:
        tuple: A tuple containing the base name (str), row (int), and column (int).
    
    Raises:
        ValueError: If the input string does not match the expected format.
    """
    match = re.match(r"(.+?)__r(-?\d+)__c(-?\d+)", baseNameWithRowCol)
    if match:
        baseName = match.group(1)
        row = int(match.group(2))
        col = int(match.group(3))
        return baseName, row, col
    else:
        raise ValueError("Input string is not in the expected format.")
    
    

def removeDuplicateBoxesRC(imageDetectionsRowCol, boundBoxChunkSize, classificationChunkSize):
    """
    Remove duplicate bounding boxes that overlap with neighboring chunks in an 11x11 grid. This is chosen because this is the
    max difference in row and column where an overlap in the image segmented could occur.
    
    Args:
        imageDetectionsRowCol (dict): Dictionary with chunked detection results, where each chunk 
                                      contains bounding boxes and corresponding confidence scores.
        boundBoxChunkSize (int): The size of each side of the bounding box image.
        classificationChunkSize (int): The size of each side of the classification image.
    
    This function directly modifies the `imageDetectionsRowCol` dictionary by removing duplicate boxes.
    """
    checkArea = math.ceil(boundBoxChunkSize / classificationChunkSize) + 1
    with tqdm(total=len(imageDetectionsRowCol), desc="Filtering crosswalks") as pbar:
        for currentKeyToFilter in imageDetectionsRowCol:
            allPointsList, allConfidenceList = imageDetectionsRowCol[currentKeyToFilter]
            toRemove = set()
            baseName, row, col = extractBaseNameAndCoords(currentKeyToFilter)
            toRemoveNeighboringMap = {}

            for dRow in range(-checkArea, checkArea + 1):  # Check 11x11 grid, with the current image being in the center.
                for dCol in range(-checkArea, checkArea + 1):
                    if dRow == 0 and dCol == 0: # If it is currently checking itself, skip to the next neighbor.
                        continue

                    newRow, newCol = row + dRow, col + dCol
                    neighboringChunk = f"{baseName}__r{newRow}__c{newCol}"
                    if neighboringChunk not in imageDetectionsRowCol:
                        continue
                    neighboringBoxes, neighboringConf = imageDetectionsRowCol[neighboringChunk]
                    toRemoveNeighboring = set()

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

            for chunk, indices in toRemoveNeighboringMap.items():
                imageDetectionsRowCol[chunk][0] = [box for i, box in enumerate(imageDetectionsRowCol[chunk][0]) if i not in indices]
                imageDetectionsRowCol[chunk][1] = [conf for i, conf in enumerate(imageDetectionsRowCol[chunk][1]) if i not in indices]

            imageDetectionsRowCol[currentKeyToFilter][0] = [box for i, box in enumerate(allPointsList) if i not in toRemove]
            imageDetectionsRowCol[currentKeyToFilter][1] = [conf for i, conf in enumerate(allConfidenceList) if i not in toRemove]

            pbar.update(1)
