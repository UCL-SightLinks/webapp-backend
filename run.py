from main import execute

# uploadDir is where the zip files are located
uploadDir = "input"

# inputType is used to determine if we are using digimap data or not
# 0 for .jpg, .jpeg, .png and their corresponding .jgw file, 1 is for .tif files
inputType = "1"

# classification threshold is used by boundBoxSegmentation to modify the threshold for the classification
classificationThreshold = 0.35

# predictionThreshold is used by prediction as a parameter to set a custom threshold for the bounding box model
predictionThreshold = 0.5

# saved is if we want to save the images of the bounding boxes
saveLabeledImage = False

# outputType is used to determine the output format
# 0 for JSON, 1 for TXT
outputType = "0"

# yoloModelType is used to determine the yolo model type
# 'n' for yolo11n-obb
# 's' for yolo11s-obb
# 'm' for yolo11m-obb
yoloModelType = 'n'

if __name__ == "__main__":
    execute(uploadDir, 
            inputType, 
            classificationThreshold, 
            predictionThreshold, 
            saveLabeledImage, 
            outputType, 
            yoloModelType
            )
