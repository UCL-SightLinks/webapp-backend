#Need to install pillow from pip, to ensure that the images are the required size
from PIL import Image
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from classificationScreening.classify import PIL_infer

def classificationSegmentation(inputFileName, classificationThreshold):
    image = Image.open(inputFileName)
    width, height = image.size
    #Classification required size
    chunkSize = 256
    listOfRowColumn = []
    # Loop to create and save chunks
    for row in range(0, height, chunkSize):
        for col in range(0, width, chunkSize):
            xDifference = 0
            yDifference = 0
            if col + chunkSize > width:
                xDifference = col + chunkSize - width
            if row + chunkSize > height:
                yDifference = row + chunkSize - height
            box = (col - xDifference, row - yDifference, col - xDifference + chunkSize, row - yDifference + chunkSize)
            cropped = image.crop(box)
            containsCrossing = PIL_infer(cropped, threshold=classificationThreshold)
            if containsCrossing:
                listOfRowColumn.append((row // chunkSize, col // chunkSize))

    return listOfRowColumn
