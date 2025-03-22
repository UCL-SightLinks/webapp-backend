import json
import os

def saveTXTOutput(outputFolder, imageName, coordinates, confidences=None):
    """Save coordinates and optional confidence scores to a TXT file with one bounding box per line"""
    txtPath = os.path.join(outputFolder, f"{imageName}.txt")
    
    with open(txtPath, 'w') as file:
        for i, coordSet in enumerate(coordinates):
            # Format each point as "lon,lat" and join with spaces
            line = " ".join([f"{point[0]},{point[1]}" for point in coordSet])
            # Add confidence score if available
            if confidences is not None and i < len(confidences):
                line += f" {confidences[i]}"
            file.write(line + "\n")

def saveToOutput(outputType, outputFolder, imageDetections):
    """Save the image detection results as either JSON or multiple TXT files"""
    if outputType == "0":
        # Save as JSON
        jsonOutput = []
        for baseName, coordAndConf in imageDetections.items():
            jsonOutput.append({
                "image": f"{baseName}",
                "coordinates": coordAndConf[0],
                "confidence": coordAndConf[1]
            })
        
        jsonPath = os.path.join(outputFolder, "output.json")
        with open(jsonPath, 'w') as file:
            json.dump(jsonOutput, file, indent=2)
        print(f"\nJSON output saved to: {jsonPath}")
    else:
        # Save as TXT files
        for baseName, coordAndConf in imageDetections.items():
            saveTXTOutput(outputFolder, baseName, coordAndConf[0], coordAndConf[1])
        print(f"\nTXT files saved to: {outputFolder}")
    
    print(f"Processed {len(imageDetections)} original images")