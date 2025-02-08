from PIL import Image
from tqdm import tqdm
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from imageSegmentation.classificationSegmentation import classificationSegmentation

def get_georef_file_path(image_path):
    """Get the corresponding georeferencing file path."""
    base, ext = os.path.splitext(image_path)
    # Try different possible georeferencing extensions
    for geo_ext in ['.jgw', '.jpgw', '.pngw', '.jpegw']:
        geo_path = base + geo_ext
        if os.path.exists(geo_path):
            return geo_path
    return None

def read_georef_data(geo_path):
    """Read georeferencing data from file."""
    try:
        with open(geo_path, 'r') as f:
            lines = f.readlines()
            if len(lines) < 6:
                raise ValueError("Incomplete georeferencing data")
            return {
                'pixelSizeX': float(lines[0].strip()),
                'rotationX': float(lines[1].strip()),
                'rotationY': float(lines[2].strip()),
                'pixelSizeY': float(lines[3].strip()),
                'topLeftXGeo': float(lines[4].strip()),
                'topLeftYGeo': float(lines[5].strip())
            }
    except Exception as e:
        raise ValueError(f"Error reading georeferencing data: {str(e)}")

def boundBoxSegmentation(classificationThreshold=0.35, outputFolder="run/output", extractDir="run/extract", progress_callback=None):
    # Create output directory if it doesn't exist
    os.makedirs(outputFolder, exist_ok=True)
    
    # Get list of image files
    image_files = [f for f in os.listdir(extractDir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    total_files = len(image_files)
    current_file = 0
    
    with tqdm(total=total_files, desc="Segmenting Images") as pbar:
        for inputFileName in image_files:
            try:
                current_file += 1
                if progress_callback:
                    progress_callback(current_file, total_files)
                    
                imagePath = os.path.join(extractDir, inputFileName)
                
                # Open and verify image
                try:
                    originalImage = Image.open(imagePath)
                    originalImage.verify()  # Verify image is valid
                    originalImage = Image.open(imagePath)  # Reopen after verify
                except Exception as e:
                    print(f"Error: Invalid image file {imagePath}: {str(e)}")
                    continue
                
                width, height = originalImage.size
                
                # Get georeferencing data
                geo_path = get_georef_file_path(imagePath)
                if not geo_path:
                    print(f"Warning: No georeferencing file found for {imagePath}")
                    continue
                
                try:
                    geo_data = read_georef_data(geo_path)
                except ValueError as e:
                    print(f"Warning: {str(e)} for {imagePath}")
                    continue
                
                # Get chunks of interest
                try:
                    chunksOfInterest = classificationSegmentation(imagePath, classificationThreshold)
                except Exception as e:
                    print(f"Error in classification for {imagePath}: {str(e)}")
                    continue
                
                # Get original filename without extension
                baseName = os.path.splitext(inputFileName)[0]
                
                # Process each chunk
                for row, col in chunksOfInterest:
                    try:
                        # Size of bounding box and classification chunks
                        largeChunkSize = 1024
                        smallChunkSize = 256
                        
                        # Calculate chunk coordinates
                        topRow = row - 1
                        topCol = col - 1
                        topX = max(0, topCol * smallChunkSize - smallChunkSize / 2)
                        topY = max(0, topRow * smallChunkSize - smallChunkSize / 2)
                        
                        # Adjust for image boundaries
                        topX = min(topX, width - largeChunkSize)
                        topY = min(topY, height - largeChunkSize)
                        
                        # Crop and save image
                        box = (int(topX), int(topY), int(topX + largeChunkSize), int(topY + largeChunkSize))
                        cropped = originalImage.crop(box)
                        
                        # Create output filenames
                        outputImage = os.path.join(outputFolder, f"{baseName}_r{row}_c{col}.jpg")
                        outputJGW = os.path.join(outputFolder, f"{baseName}_r{row}_c{col}.jgw")
                        
                        # Save image and georeferencing data
                        cropped.save(outputImage, quality=95)
                        with open(outputJGW, 'w') as file:
                            file.write(f"{geo_data['pixelSizeX']:.10f}\n")
                            file.write(f"{geo_data['rotationX']:.10f}\n")
                            file.write(f"{geo_data['rotationY']:.10f}\n")
                            file.write(f"{geo_data['pixelSizeY']:.10f}\n")
                            file.write(f"{(geo_data['topLeftXGeo'] + topX * geo_data['pixelSizeX']):.10f}\n")
                            file.write(f"{(geo_data['topLeftYGeo'] + topY * geo_data['pixelSizeY']):.10f}\n")
                        
                        pbar.update(1)
                    except Exception as e:
                        print(f"Error processing chunk row={row}, col={col} for {imagePath}: {str(e)}")
                        continue
                        
            except Exception as e:
                print(f"Error processing {imagePath}: {str(e)}")
                continue