from PIL import Image
from tqdm import tqdm
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from imageSegmentation.classificationSegmentation import classificationSegmentation

def boundBoxSegmentation(classificationThreshold=0.35, outputFolder = "run/output", extractDir = "run/extract"):
    with tqdm(total=(len(os.listdir(extractDir))//2), desc="Segmenting Images") as pbar:
        for inputFileName in os.listdir(extractDir):
            if inputFileName.endswith(('.png', '.jpg', '.jpeg')):
                try:
                    imagePath = os.path.join(extractDir, inputFileName)
                    originalImage = Image.open(imagePath)
                    width, height = originalImage.size
                    #Size of bounding box chunks
                    largeChunkSize = 1024
                    #Size of classification chunks
                    smallChunkSize = 256
                    #chunksOfInterest will be retrieved from kostas' program.
                    chunksOfInterest = classificationSegmentation(imagePath, classificationThreshold)
                    
                    # Get original filename without extension
                    baseName = os.path.splitext(inputFileName)[0]
                    os.makedirs(outputFolder, exist_ok=True)

                    #data for georeferencing
                    with open(imagePath.replace('jpg', 'jgw'), 'r') as jgwFile:
                        lines = jgwFile.readlines()
                    pixelSizeX = float(lines[0].strip())
                    rotationX = float(lines[1].strip())
                    rotationY = float(lines[2].strip())
                    pixelSizeY = float(lines[3].strip())
                    topLeftXGeo = float(lines[4].strip())
                    topLeftYGeo = float(lines[5].strip())

                    for row, col in chunksOfInterest:
                        #setting cases when point of interest is at the edges
                        topRow = row - 1
                        topCol = col - 1
                        topX = topCol * smallChunkSize - smallChunkSize / 2 if topCol > 0 else 0 #This is the top left x
                        topY = topRow * smallChunkSize - smallChunkSize / 2 if topRow > 0 else 0 #This is the top left y

                        #setting case if there are overlaps in the image
                        if topX + largeChunkSize > width:
                            topX = width - largeChunkSize
                        if topY + largeChunkSize > height:
                            topY = height - largeChunkSize

                        box = (topX, topY, topX + largeChunkSize, topY + largeChunkSize)
                        cropped = originalImage.crop(box)
                        
                        # Create output filenames using original name plus row and column
                        outputImage = f"{outputFolder}/{baseName}_r{row}_c{col}.jpg"
                        outputJGW = f"{outputFolder}/{baseName}_r{row}_c{col}.jgw"
                        
                        cropped.save(outputImage)
                        with open(outputJGW, 'w') as file:
                            file.write(f"{pixelSizeX:.10f}\n")
                            file.write(f"{rotationX:.10f}\n")
                            file.write(f"{rotationY:.10f}\n")
                            file.write(f"{pixelSizeY:.10f}\n")
                            file.write(f"{(topLeftXGeo + topX * pixelSizeX):.10f}\n")
                            file.write(f"{(topLeftYGeo + topY * pixelSizeY):.10f}\n")
                        pbar.update(1)
                except Exception as e:
                    print(f"Error opening {imagePath}: {e}")