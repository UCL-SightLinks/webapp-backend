from PIL import Image
from osgeo import gdal
from tqdm import tqdm
import os
import sys

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
    chunks in the center when possible.

    During the segmentation, we are also updating the georeferencing data for the new TIF file, keeping the TIF format.

    Args:
        classificationThreshold (float): The threshold for the classification model.
        extractDir (str): The path to the directory where all of the input images are.
        boundBoxChunkSize (int): The size of each side of the bounding box image.
        classificationChunkSize (int): The size of each side of the classification image.
    Returns:
        imageAndDatas (list): A list of the input image name, segmented TIF image, row, and column.
    """
    with tqdm(total=(len(os.listdir(extractDir))), desc="Segmenting Images") as pbar:
        imageAndDatas = []
        chunkSeen = set()
        for inputFileName in os.listdir(extractDir):
            if inputFileName.endswith(('.tif')):
                try:
                    imagePath = os.path.join(extractDir, inputFileName)
                    dataset = gdal.Open(imagePath, gdal.GA_ReadOnly)
                    if dataset is None:
                        raise Exception(f"Failed to open {imagePath}")
                    
                    width = dataset.RasterXSize
                    height = dataset.RasterYSize
                    # Get the georeference data (this will be used to preserve georeferencing)
                    geoTransform = dataset.GetGeoTransform()
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
                        georeferencedTopX = geoTransform[0] + topX * geoTransform[1] + topY * geoTransform[2]
                        georeferencedTopY = geoTransform[3] + topX * geoTransform[4] + topY * geoTransform[5]
                        
                        # Use GDAL to create the cropped image, preserving georeference
                        imageChunk = f"{inputFileName}{(topX, topY, boundBoxChunkSize)}"
                        if imageChunk in chunkSeen:
                            continue
                        chunkSeen.add(imageChunk)
                        cropped = gdal.Translate("", dataset, srcWin=[topX, topY, boundBoxChunkSize, boundBoxChunkSize], 
                                    projWin=[georeferencedTopX, georeferencedTopY, geoTransform[0] + (topX + boundBoxChunkSize) * geoTransform[1], geoTransform[3] + (topY + boundBoxChunkSize) * geoTransform[5]], 
                                    format="MEM")
                        imageAndDatas.append((inputFileName, cropped, row, col))
                except Exception as e:
                    print(f"Error opening {imagePath}: {e}")
            pbar.update(1)
        return imageAndDatas