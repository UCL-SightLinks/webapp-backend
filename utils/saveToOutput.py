import json
import os

def saveTXTOutput(outputFolder, imageName, coordinates, confidences=None):
    """Save coordinates and optional confidence scores to a TXT file with one bounding box per line"""
    # Don't create empty files if no coordinates
    if not coordinates or len(coordinates) == 0:
        return False
        
    txtPath = os.path.join(outputFolder, f"{imageName}.txt")
    
    with open(txtPath, 'w') as file:
        for i, coordSet in enumerate(coordinates):
            # Format each point as "lon,lat" and join with spaces
            line = " ".join([f"{point[0]},{point[1]}" for point in coordSet])
            # Add confidence score if available
            if confidences is not None and i < len(confidences):
                line += f" {confidences[i]}"
            file.write(line + "\n")
    
    return True

def saveToOutput(outputType, outputFolder, imageDetections):
    """Save the image detection results as either JSON or multiple TXT files
    
    Returns:
        bool: True if detections were found and saved, False if no detections were found
    """
    
    # Check if there are any detections at all
    has_detections = False
    total_detections = 0
    
    for baseName, coordAndConf in imageDetections.items():
        if coordAndConf and coordAndConf[0] and len(coordAndConf[0]) > 0:
            has_detections = True
            total_detections += len(coordAndConf[0])
    
    # Log detection status
    print(f"Total detections found: {total_detections}")
    
    # If no detections were found, create a marker file but don't generate output files
    if not has_detections or total_detections == 0:
        no_detection_marker = os.path.join(outputFolder, "no_detections.txt")
        with open(no_detection_marker, 'w') as f:
            f.write("No detections were found in any of the images.")
        print(f"No detections found. Created marker file at {no_detection_marker}")
        return False
    
    if outputType == "0":
        # Save as JSON
        jsonOutput = []
        for baseName, coordAndConf in imageDetections.items():
            # Only add images with actual detections
            if coordAndConf[0] and len(coordAndConf[0]) > 0:
                jsonOutput.append({
                    "image": f"{baseName}",
                    "coordinates": coordAndConf[0],
                    "confidence": coordAndConf[1]
                })
        
        # Double check that we have actual data to save
        if not jsonOutput:
            print("No data to save in JSON output")
            no_detection_marker = os.path.join(outputFolder, "no_detections.txt")
            with open(no_detection_marker, 'w') as f:
                f.write("No detections were found in any of the images.")
            return False
            
        # Use a more descriptive name for the JSON output file
        jsonPath = os.path.join(outputFolder, "detections.json")
        with open(jsonPath, 'w') as file:
            json.dump(jsonOutput, file, indent=2)
        print(f"\nJSON output saved to: {jsonPath}")
        
    else:
        # Save as TXT files
        files_created = 0
        for baseName, coordAndConf in imageDetections.items():
            if saveTXTOutput(outputFolder, baseName, coordAndConf[0], coordAndConf[1]):
                files_created += 1
        
        # Check if any files were created
        if files_created == 0:
            print("No TXT files were created - no detections found")
            no_detection_marker = os.path.join(outputFolder, "no_detections.txt")
            with open(no_detection_marker, 'w') as f:
                f.write("No detections were found in any of the images.")
            return False
            
        print(f"\nCreated {files_created} TXT files in: {outputFolder}")
    
    print(f"Processed {len(imageDetections)} original images with {total_detections} detections")
    return True