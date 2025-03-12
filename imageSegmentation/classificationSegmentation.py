from PIL import Image
import math
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from classificationScreening.classify import PIL_infer

def classificationSegmentation(inputFileName, classificationThreshold, classificationChunkSize):
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
    image = Image.open(inputFileName)
    width, height = image.size
    listOfRowCol = []
    # row and col represents the coordinates for the top left point of the new cropped image
    for row in range(0, height, classificationChunkSize):
        for col in range(0, width, classificationChunkSize):
            xDifference = 0
            yDifference = 0
            if col + classificationChunkSize > width:
                xDifference = col + classificationChunkSize - width
            if row + classificationChunkSize > height:
                yDifference = row + classificationChunkSize - height
            box = (col - xDifference, row - yDifference, col - xDifference + classificationChunkSize, row - yDifference + classificationChunkSize)
            cropped = image.crop(box)
            containsCrossing = PIL_infer(cropped, threshold=classificationThreshold)
            if containsCrossing:
                rowToAdd = row // classificationChunkSize
                colToAdd = col // classificationChunkSize
                # The reason we are adding this is due to the way it is being set up, it allows us to filter the output more efficiently.
                # Changing the row and column of interest of these will not affect the image segmentation at all, so there should be no
                # unexpected behavior.
                if xDifference:
                    colToAdd = math.ceil(width / classificationChunkSize) - 2
                elif col == 0:
                    colToAdd = 1
                if yDifference:
                    rowToAdd = math.ceil(height / classificationChunkSize) - 2
                elif row == 0:
                    rowToAdd = 1
                listOfRowCol.append((rowToAdd, colToAdd))

    return listOfRowCol