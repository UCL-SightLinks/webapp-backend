from PIL import Image
from osgeo import gdal
from tqdm import tqdm
import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from imageSegmentation.classificationSegmentation import classificationSegmentation

def boundBoxSegmentationJGW(classificationThreshold=0.35, extractDir = "run/extract", boundBoxChunkSize=1024, classificationChunkSize=256):
    """
    This function will iterate through all of the .png, .jpg, and .jpeg images from the extract directory.
    It will then call the classificationSegmentation function and receive all the chunks of interest for each image.
    From these chunks of interest, it will resegment them into boxes with size boundBoxChunkSize, with the original
    chunks in the center when possible.

    After resegmenting the image, it will calculate the new georeferencing data, including the new topLeftXGeo, 
    and topLeftYGeo. It will then save this new segmented image, and all georeferencing data needed for this chunk in
    imageAndDatas. Once all of the image has been processed, imageAndDatas is returned.

    Args:
        classificationThreshold (float): The threshold for the classification model.
        extractDir (str): The path to the directory where all of the input images are.
        boundBoxChunkSize (int): The size of each side of the bounding box image.
        classificationChunkSize (int): The size of each side of the classification image.
    
    Returns:
        imageAndDatas (list): A list of the input image name, segmented image, georeferencing data, row, and column.
    """
    # Ensure classificationThreshold is a float
    classificationThreshold = float(classificationThreshold)
    
    with tqdm(total=(len(os.listdir(extractDir))//2), desc="Segmenting Images") as pbar:
        imageAndDatas = []
        chunkSeen = set()
        for inputFileName in os.listdir(extractDir):
            if inputFileName.endswith(('.png', '.jpg', '.jpeg')):
                try:
                    imagePath = os.path.join(extractDir, inputFileName)
                    originalImage = Image.open(imagePath)
                    width, height = originalImage.size
                    chunksOfInterest = classificationSegmentation(inputFileName=imagePath, classificationThreshold=classificationThreshold, classificationChunkSize=classificationChunkSize)
                    #data for georeferencing
                    baseName, _ = os.path.splitext(imagePath)
                    jgwPath = baseName + ".jgw"
                    with open(jgwPath) as jgwFile:
                        lines = jgwFile.readlines()
                    pixelSizeX = float(lines[0].strip())
                    pixelSizeY = float(lines[3].strip())
                    topLeftXGeo = float(lines[4].strip())
                    topLeftYGeo = float(lines[5].strip())

                    for row, col in chunksOfInterest:
                        offset = (boundBoxChunkSize - classificationChunkSize) / 2
                        topX = col * classificationChunkSize - offset if col * classificationChunkSize - offset > 0 else 0
                        topY = row * classificationChunkSize - offset if row * classificationChunkSize - offset > 0 else 0

                        if topX + boundBoxChunkSize > width:
                            topX = width - boundBoxChunkSize
                        if topY + boundBoxChunkSize > height:
                            topY = height - boundBoxChunkSize

                        box = (topX, topY, topX + boundBoxChunkSize, topY + boundBoxChunkSize)
                        imageChunk = f"{inputFileName}{(topX, topY, boundBoxChunkSize)}"
                        if imageChunk in chunkSeen:
                            continue
                        chunkSeen.add(imageChunk)
                        cropped = originalImage.crop(box)

                        topLeftXGeoInterest = topLeftXGeo + topX * pixelSizeX
                        topLeftYGeoInterest = topLeftYGeo + topY * pixelSizeY
                        imageAndDatas.append((inputFileName, cropped, pixelSizeX, pixelSizeY, topLeftXGeoInterest, topLeftYGeoInterest, row, col)) 
                except Exception as e:
                    print(f"Error opening {imagePath}: {e}")
            pbar.update(1)
        return imageAndDatas


def boundBoxSegmentationTIF(classificationThreshold=0.35, extractDir = "run/extract", boundBoxChunkSize=1024, classificationChunkSize=256):
    """
    This function will iterate through all of the .tif images from the extract directory.
    It will then call the classificationSegmentation function and receive all the chunks of interest for each image.
    From these chunks of interest, it will resegment them into boxes with size boundBoxChunkSize, with the original
    chunks being at the centre of these boxes.

    Args:
        classificationThreshold (float): The threshold for the classification model.
        extractDir (str): The directory that the images are in.
        boundBoxChunkSize (int): The size of each side of the bounding box image.
        classificationChunkSize (int): The size of each side of the classification image.
    
    Returns:
        imageAndDatas (list): A list of the input image name, segmented image, georeferencing data, row, and column.
    """
    # Ensure classificationThreshold is a float
    classificationThreshold = float(classificationThreshold)
    
    with tqdm(total=len([f for f in os.listdir(extractDir) if f.endswith('.tif') or f.endswith('.tiff')]), desc="Segmenting Images") as pbar:
        imageAndDatas = []
        for inputFileName in os.listdir(extractDir):
            if inputFileName.endswith(('.tif', '.tiff')):
                try:
                    imagePath = os.path.join(extractDir, inputFileName)
                    dataset = gdal.Open(imagePath)
                    band = dataset.GetRasterBand(1)
                    # Convert GDAL dataset to PIL Image
                    data = band.ReadAsArray()
                    if dataset.RasterCount == 3:  # RGB
                        # Get the other bands
                        band2 = dataset.GetRasterBand(2)
                        band3 = dataset.GetRasterBand(3)
                        data2 = band2.ReadAsArray()
                        data3 = band3.ReadAsArray()
                        # Stack them together
                        data = np.dstack((data, data2, data3))
                        image = Image.fromarray(data.astype(np.uint8))
                    else:  # Use as grayscale
                        image = Image.fromarray(data.astype(np.uint8))
                    
                    width, height = image.size
                    # Get the geotransform data (for georeferencing)
                    geotransform = dataset.GetGeoTransform()
                    pixelSizeX = geotransform[1]
                    pixelSizeY = abs(geotransform[5])  # Absolute value because it's negative
                    topLeftXGeo = geotransform[0]
                    topLeftYGeo = geotransform[3]
                    
                    chunksOfInterest = classificationSegmentation(inputFileName=imagePath, classificationThreshold=classificationThreshold, classificationChunkSize=classificationChunkSize)

                    for row, col in chunksOfInterest:
                        offset = (boundBoxChunkSize - classificationChunkSize) / 2
                        topX = col * classificationChunkSize - offset if col * classificationChunkSize - offset > 0 else 0
                        topY = row * classificationChunkSize - offset if row * classificationChunkSize - offset > 0 else 0
                        
                        if topX + boundBoxChunkSize > width:
                            topX = width - boundBoxChunkSize
                        if topY + boundBoxChunkSize > height:
                            topY = height - boundBoxChunkSize
                        # Convert the pixel coordinates to georeferenced coordinates
                        georeferencedTopX = geotransform[0] + topX * geotransform[1] + topY * geotransform[2]
                        georeferencedTopY = geotransform[3] + topX * geotransform[4] + topY * geotransform[5]
                        
                        # Use GDAL to create the cropped image, preserving georeference
                        imageChunk = f"{inputFileName}{(topX, topY, boundBoxChunkSize)}"
                        if imageChunk in chunkSeen:
                            continue
                        chunkSeen.add(imageChunk)
                        cropped = gdal.Translate("", dataset, srcWin=[topX, topY, boundBoxChunkSize, boundBoxChunkSize], 
                                    projWin=[georeferencedTopX, georeferencedTopY, geotransform[0] + (topX + boundBoxChunkSize) * geotransform[1], geotransform[3] + (topY + boundBoxChunkSize) * geotransform[5]], 
                                    format="MEM")
                        imageAndDatas.append((inputFileName, cropped, row, col))
                except Exception as e:
                    print(f"Error opening {imagePath}: {e}")
            pbar.update(1)
        return imageAndDatas