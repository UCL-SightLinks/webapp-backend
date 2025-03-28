from PIL import Image
from osgeo import gdal
from tqdm import tqdm
import os
import sys
import traceback

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
                    chunksOfInterest = classificationSegmentation(inputFileName=imagePath, classificationThreshold=classificationThreshold, classificationChunkSize=classificationChunkSize, boundBoxChunkSize=boundBoxChunkSize)
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
    print("\n=== Starting TIF Segmentation ===")
    print(f"Extract directory: {extractDir}")
    print(f"Classification threshold: {classificationThreshold}")
    print(f"Bound box chunk size: {boundBoxChunkSize}")
    print(f"Classification chunk size: {classificationChunkSize}")
    
    with tqdm(total=(len(os.listdir(extractDir))), desc="Segmenting Images") as pbar:
        imageAndDatas = []
        chunkSeen = set()
        for inputFileName in os.listdir(extractDir):
            if inputFileName.lower().endswith(('.tif', '.tiff')):
                try:
                    imagePath = os.path.join(extractDir, inputFileName)
                    print(f"\nProcessing TIF file: {inputFileName}")
                    print(f"Full path: {imagePath}")
                    
                    # Open the TIF file
                    dataset = gdal.Open(imagePath, gdal.GA_ReadOnly)
                    if dataset is None:
                        raise Exception(f"Failed to open {imagePath}")
                    
                    width = dataset.RasterXSize
                    height = dataset.RasterYSize
                    num_bands = dataset.RasterCount
                    
                    # Log TIF file details
                    print(f"TIF file details:")
                    print(f"- Dimensions: {width}x{height}")
                    print(f"- Number of bands: {num_bands}")
                    print(f"- Projection: {dataset.GetProjection()}")
                    print(f"- Geotransform: {dataset.GetGeoTransform()}")
                    
                    # Get the georeference data
                    geoTransform = dataset.GetGeoTransform()
                    
                    # Run classification segmentation
                    print("\nRunning classification segmentation...")
                    chunksOfInterest = classificationSegmentation(
                        inputFileName=imagePath,
                        classificationThreshold=classificationThreshold,
                        classificationChunkSize=classificationChunkSize,
                        boundBoxChunkSize=boundBoxChunkSize
                    )
                    
                    print(f"Found {len(chunksOfInterest)} chunks of interest")
                    
                    if not chunksOfInterest:
                        print("No chunks of interest found in the image")
                        continue
                    
                    # Process each chunk of interest
                    for row, col in chunksOfInterest:
                        try:
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
                            
                            # Create unique identifier for this chunk
                            imageChunk = f"{inputFileName}{(topX, topY, boundBoxChunkSize)}"
                            if imageChunk in chunkSeen:
                                continue
                            chunkSeen.add(imageChunk)
                            
                            print(f"\nProcessing chunk at row {row}, col {col}")
                            print(f"Pixel coordinates: ({topX}, {topY})")
                            print(f"Georeferenced coordinates: ({georeferencedTopX}, {georeferencedTopY})")
                            
                            # Create a memory dataset for the cropped image
                            cropped = gdal.GetDriverByName('MEM').Create('', boundBoxChunkSize, boundBoxChunkSize, num_bands)
                            
                            # Copy geotransform and projection
                            cropped.SetGeoTransform([
                                georeferencedTopX,
                                geoTransform[1],
                                geoTransform[2],
                                georeferencedTopY,
                                geoTransform[4],
                                geoTransform[5]
                            ])
                            cropped.SetProjection(dataset.GetProjection())
                            
                            # Copy data from source to cropped image
                            for b in range(num_bands):
                                src_band = dataset.GetRasterBand(b + 1)
                                dst_band = cropped.GetRasterBand(b + 1)
                                src_band.ReadRaster(topX, topY, boundBoxChunkSize, boundBoxChunkSize,
                                                  buf_type=src_band.DataType,
                                                  buf_obj=dst_band)
                            
                            baseName, _ = os.path.splitext(inputFileName)
                            imageAndDatas.append((baseName, cropped, row, col))
                            print(f"Successfully added chunk to results")
                            
                        except Exception as chunk_error:
                            print(f"Error processing chunk at row {row}, col {col}: {str(chunk_error)}")
                            print(traceback.format_exc())
                            continue
                    
                except Exception as e:
                    print(f"Error processing TIF file {inputFileName}: {str(e)}")
                    print(traceback.format_exc())
            pbar.update(1)
            
        print(f"\n=== TIF Segmentation Complete ===")
        print(f"Total chunks created: {len(imageAndDatas)}")
        return imageAndDatas