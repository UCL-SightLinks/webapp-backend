from PIL import Image
import math
import os
import sys
from osgeo import gdal
import numpy as np
import torchvision.transforms.functional as TF
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from classificationScreening.classify import PIL_infer

def classificationSegmentation(inputFileName, classificationThreshold, classificationChunkSize, boundBoxChunkSize):
    """
    Divides the images into square chunks, and passes it into the classification model.
    It will then keep track of the row and column where the classification model returns true, and return it.

    Args:
        inputFileName (str): The name of the file we are trying to open.
        classificationThreshold (float): The threshold for the classification model.
        classificationChunkSize (int): The size of chunks we are breaking down the original image to.
    
    Returns:
        listOfRowCol (list): A list of row and columns of interest.
    """
    print("\n=== Starting Classification Segmentation ===")
    print(f"Input file: {inputFileName}")
    print(f"Classification threshold: {classificationThreshold}")
    print(f"Classification chunk size: {classificationChunkSize}")
    print(f"Bound box chunk size: {boundBoxChunkSize}")
    
    # Check if the file is a TIF
    if inputFileName.lower().endswith(('.tif', '.tiff')):
        print("\nProcessing TIF file...")
        try:
            # Try to open with PIL directly first
            print("Attempting to open TIF with PIL...")
            image = Image.open(inputFileName)
            width, height = image.size
            print(f"Successfully opened TIF with PIL")
            print(f"TIF dimensions: {width}x{height}")
            print(f"TIF mode: {image.mode}")
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                print(f"Converting from {image.mode} to RGB")
                image = image.convert('RGB')
                
        except Exception as e:
            print(f"Error opening TIF with PIL: {str(e)}")
            print("Falling back to GDAL...")
            
            # Open with GDAL for TIF files if PIL fails
            dataset = gdal.Open(inputFileName, gdal.GA_ReadOnly)
            if dataset is None:
                raise Exception(f"Failed to open TIF file: {inputFileName}")
                
            width = dataset.RasterXSize
            height = dataset.RasterYSize
            num_bands = dataset.RasterCount
            
            print(f"TIF file details:")
            print(f"- Dimensions: {width}x{height}")
            print(f"- Number of bands: {num_bands}")
            print(f"- Projection: {dataset.GetProjection()}")
            print(f"- Geotransform: {dataset.GetGeoTransform()}")
            
            # Alternative approach without using ReadAsArray
            print("\nReading TIF data using alternative method...")
            
            # Create a temporary file to convert with GDAL
            temp_jpg = f"{os.path.splitext(inputFileName)[0]}_temp.jpg"
            try:
                # Use gdal.Translate to convert the TIF to a format PIL can read
                print(f"Converting TIF to temporary JPG: {temp_jpg}")
                gdal.Translate(temp_jpg, dataset, format="JPEG")
                
                # Open the temporary file with PIL
                print(f"Opening converted JPG with PIL")
                image = Image.open(temp_jpg)
                width, height = image.size
                print(f"Temporary JPG dimensions: {width}x{height}")
                
                # Cleanup
                dataset = None
            except Exception as inner_e:
                print(f"Error in GDAL alternative method: {str(inner_e)}")
                raise Exception(f"Could not process TIF file: {str(inner_e)}")
            finally:
                # Clean up temporary file
                if os.path.exists(temp_jpg):
                    try:
                        os.remove(temp_jpg)
                        print(f"Removed temporary JPG file")
                    except:
                        print(f"Could not remove temporary file: {temp_jpg}")
    else:
        print("\nProcessing non-TIF file...")
        # Use PIL for non-TIF files
        image = Image.open(inputFileName)
        width, height = image.size
        print(f"Image dimensions: {width}x{height}")

    # Ensure image is in RGB mode
    if image.mode != 'RGB':
        print(f"Converting image from {image.mode} to RGB mode")
        image = image.convert('RGB')

    lowerFilteringBound = math.ceil(boundBoxChunkSize / classificationChunkSize) // 2 - 1
    upperWidthFilteringBound = math.ceil(width / classificationChunkSize) - 1 - lowerFilteringBound
    upperHeightFilteringBound = math.ceil(height / classificationChunkSize) - 1 - lowerFilteringBound
    
    print("\nProcessing image chunks...")
    print(f"Lower filtering bound: {lowerFilteringBound}")
    print(f"Upper width filtering bound: {upperWidthFilteringBound}")
    print(f"Upper height filtering bound: {upperHeightFilteringBound}")
    
    listOfRowCol = []
    total_chunks = 0
    chunks_with_crossings = 0
    
    # row and col represents the coordinates for the top left point of the new cropped image
    for row in range(0, height, classificationChunkSize):
        for col in range(0, width, classificationChunkSize):
            total_chunks += 1
            xDifference = 0
            yDifference = 0
            if col + classificationChunkSize > width:
                xDifference = col + classificationChunkSize - width
            if row + classificationChunkSize > height:
                yDifference = row + classificationChunkSize - height
            box = (col - xDifference, row - yDifference, col - xDifference + classificationChunkSize, row - yDifference + classificationChunkSize)
            cropped = image.crop(box)
            
            # Ensure cropped image is in RGB mode and has the correct size
            if cropped.mode != 'RGB':
                cropped = cropped.convert('RGB')
            if cropped.size != (classificationChunkSize, classificationChunkSize):
                cropped = cropped.resize((classificationChunkSize, classificationChunkSize), Image.Resampling.LANCZOS)
            
            # Convert to tensor and normalize
            tensor_im = TF.pil_to_tensor(cropped).float() / 255.0
            
            # Check if the image contains a crossing
            containsCrossing = PIL_infer(cropped, threshold=classificationThreshold)
            
            if containsCrossing:
                chunks_with_crossings += 1
                rowToAdd = row // classificationChunkSize
                colToAdd = col // classificationChunkSize

                if colToAdd >= upperWidthFilteringBound:
                    colToAdd = upperWidthFilteringBound
                elif colToAdd <= lowerFilteringBound:
                    colToAdd = lowerFilteringBound
                if rowToAdd >= upperHeightFilteringBound:
                    rowToAdd = upperHeightFilteringBound
                elif rowToAdd <= lowerFilteringBound:
                    rowToAdd = lowerFilteringBound
                listOfRowCol.append((rowToAdd, colToAdd))
                print(f"Found crossing at row {rowToAdd}, col {colToAdd}")

    print(f"\nClassification Segmentation Complete:")
    print(f"Total chunks processed: {total_chunks}")
    print(f"Chunks with crossings: {chunks_with_crossings}")
    print(f"Total regions of interest: {len(listOfRowCol)}")
    
    return listOfRowCol