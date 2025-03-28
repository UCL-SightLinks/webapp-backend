from PIL import Image
from osgeo import gdal
from tqdm import tqdm
import os
import sys
import traceback
import shutil

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
                    
                    # Try to open with PIL first
                    try:
                        print("Attempting to open TIF with PIL...")
                        pilImage = Image.open(imagePath)
                        width, height = pilImage.size
                        print(f"PIL opened TIF file with dimensions: {width}x{height}")
                        
                        # Open with GDAL just to get georeferencing data
                        dataset = gdal.Open(imagePath, gdal.GA_ReadOnly)
                        if dataset is None:
                            raise Exception(f"Failed to open {imagePath} with GDAL")
                        
                        # Get georeferencing data
                        geoTransform = dataset.GetGeoTransform()
                        projection = dataset.GetProjection()
                        
                        # Log TIF file details
                        print(f"TIF file details:")
                        print(f"- Dimensions: {width}x{height}")
                        print(f"- PIL mode: {pilImage.mode}")
                        print(f"- Projection: {projection}")
                        print(f"- Geotransform: {geoTransform}")
                        
                        # Create a temporary JPEG file for each segment later
                        temp_dir = os.path.join(os.path.dirname(imagePath), "temp_segments")
                        os.makedirs(temp_dir, exist_ok=True)
                        
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
                            # Clean up temporary directory
                            try:
                                shutil.rmtree(temp_dir)
                                print(f"Removed temporary directory: {temp_dir}")
                            except:
                                print(f"Failed to remove temporary directory: {temp_dir}")
                            continue
                        
                        # Process each chunk of interest
                        for row, col in chunksOfInterest:
                            try:
                                offset = (boundBoxChunkSize - classificationChunkSize) / 2
                                topX = col * classificationChunkSize - offset if col * classificationChunkSize - offset > 0 else 0
                                topY = row * classificationChunkSize - offset if row * classificationChunkSize - offset > 0 else 0
                                
                                topX = int(topX)
                                topY = int(topY)
                                
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
                                
                                # Crop the image with PIL
                                box = (topX, topY, topX + boundBoxChunkSize, topY + boundBoxChunkSize)
                                print(f"Cropping box: {box}")
                                cropped_pil = pilImage.crop(box)
                                
                                # Save to temporary file
                                temp_jpg = os.path.join(temp_dir, f"segment_{row}_{col}.jpg")
                                cropped_pil.save(temp_jpg)
                                print(f"Saved temporary segment to {temp_jpg}")
                                
                                # Create a GDAL dataset with the right georeferencing
                                driver = gdal.GetDriverByName('MEM')
                                num_bands = len(cropped_pil.getbands()) 
                                print(f"Creating GDAL dataset with {num_bands} bands")
                                
                                # We're going to create a simpler version of the dataset
                                # that preserves the geolocation but uses the PIL image data
                                gdal_dataset = driver.Create('', boundBoxChunkSize, boundBoxChunkSize, num_bands)
                                
                                # Set the georeferencing information
                                cropped_geo_transform = [
                                    georeferencedTopX,
                                    geoTransform[1],
                                    geoTransform[2],
                                    georeferencedTopY,
                                    geoTransform[4],
                                    geoTransform[5]
                                ]
                                gdal_dataset.SetGeoTransform(cropped_geo_transform)
                                gdal_dataset.SetProjection(projection)
                                
                                print(f"Georeferencing set for cropped dataset")
                                
                                baseName, _ = os.path.splitext(inputFileName)
                                imageAndDatas.append((baseName, gdal_dataset, row, col))
                                print(f"Successfully added chunk to results")
                                
                            except Exception as chunk_error:
                                print(f"Error processing chunk at row {row}, col {col}: {str(chunk_error)}")
                                print(traceback.format_exc())
                                continue
                        
                        # Clean up temporary directory
                        try:
                            shutil.rmtree(temp_dir)
                            print(f"Removed temporary directory: {temp_dir}")
                        except:
                            print(f"Failed to remove temporary directory: {temp_dir}")
                                
                    except Exception as pil_error:
                        print(f"Error processing with PIL: {str(pil_error)}")
                        print(traceback.format_exc())
                        print("Cannot proceed with TIF processing due to PIL error")
                        continue
                    
                except Exception as e:
                    print(f"Error processing TIF file {inputFileName}: {str(e)}")
                    print(traceback.format_exc())
            pbar.update(1)
            
        print(f"\n=== TIF Segmentation Complete ===")
        print(f"Total chunks created: {len(imageAndDatas)}")
        return imageAndDatas