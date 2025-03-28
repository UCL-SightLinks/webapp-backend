import os
import yaml
from pathlib import Path
import shutil
from PIL import Image
import math
import numpy as np
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as PltPolygon
import random

zebra_labels = "zebra_annotations/txt_annotations"
zebra_images = "zebra_annotations/zebra_images"
save_dir = "zebra_annotations/classification_data"
segments = 4

# Each box has format (x_1, y_1, x_2, y_2) - this does mean this is an estimate.
# box_1 is the bounding box of the crosswalk, and box_2 is that of the tile.
def check_box_intersection(box_1, box_2, threshold=0.6):
    """
    A variant of Intersection-over-Union calculations that takes into consideration the relative scale
    difference between the two input boxes by scaling the smaller bounding box by the difference factor.
    Otherwise the calculation process is the same

    Args:
        box_1 (float tuple): A pair of (x, y) coordinates forming a non-zero area box (does not have to be regular)
        box_2 (float tuple): A pair of (x, y) coordinates forming a non-zero area box (does not have to be regular)
        thresold (float): The minimum scaled IoU value required for the boxes to be classified as intersecting

    Returns:
        intersection (boolean): Whether the boxes overlap according to the given threshold
    """

    # Threshold is the min IoU required to consider it as having a crosswalk - the minimum percent area of the crosswalk that must be in the tile
    formatted_box_1 = [[box_1[0], box_1[1]], [box_1[2], box_1[1]], [box_1[2], box_1[3]], [box_1[0], box_1[3]]]  # Formatting follows shapely clockwise system

    poly_1 = Polygon(formatted_box_1)
    poly_2 = Polygon(box_2)
    try:
        iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
        scaled_iou = iou * ((poly_1.area) / (poly_2.area))
        # print("--------------")
        # print(iou, scaled_iou)
        return (scaled_iou > threshold)
    except ZeroDivisionError:
        # In the case of a zero division error
        print("ZERO DIVISION ERROR", poly_1.area, poly_2.area)
        return False

def load_yaml_database(yaml_path):
    """
    Gives the structure of a YAML database, required for loading in new YOLO datasets - which we do not do in
    our case, but can be useful for importing your own datasets.

    Args:
        yaml_path (string): The path to the yaml database configuration file

    Returns:
        (train_dir, valid_dir, test_dir), (image_size, classes):
            train_dir (string): The path to the directory containing the training data
            valid_dir (string): The path to the directory containing the validation data
            test_dir (string): The path to the directory containing the testing data
            image_size (int array): The dimensions of the image in terms of width, height, channels, etc.
            classes (string array): The names of the different classes represented within the dataset.
    
    """
    config_file = None

    with open(yaml_path, "r") as file:
        config_file = yaml.safe_load(file)
    
    root = Path(config_file['path'])
    train_dir = root / config_file['train']  # Training data
    valid_dir = root / config_file['val']  # Validation data
    test_dir = root / config_file['test']  # Testing data
    image_size = config_file['img_size']
    classes = config_file['names']

    # Returns in format (directories, label_description)
    return (train_dir, valid_dir, test_dir), (image_size, classes)

def convert_database_to_segments(image_dir, label_dir, dst_dir, overwrite=False):
    """
    Takes in an annotated dataset in the format described in the data annotation section (which can be
    found on our website) labelled for bounding box detection, and segmented the images into smaller images
    with labels for classification training. 
    
    Data is saved directly into the target directory as png files, with the associated labels being saved 
    directly as well as txt files with the same name as its associated image. All data is saved with a name 
    corresponding to its index.
    
    Args:
        image_dir (string): The path to the directory containing the image data (must be stored as one 
            of the following format: ('.jpg', '.jpeg', '.png') )
        label_dir (string): The path to the directory containing the label data (must be stored in text)
        dst_dir (string): The path to the directory where the data will be stored - will be overwritten
            if that setting has been enabled, and will be created if it doesn't exist yet.
        overwrite (boolean): Whether to delete the current data
    
    Returns:
        None
    
    """

    if not os.path.exists(image_dir) or not os.path.exists(label_dir):
        print("Error: Image or label directories do not exist")
        return
    
    # If there isn't a directory created to save the new data to, create it
    dst_dir = Path(dst_dir)
    try:
        if overwrite and dst_dir.exists() and dst_dir.is_dir():
            shutil.rmtree(dst_dir)
            dst_dir.mkdir(parents=True, exist_ok=True)
        else:
            dst_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        # If the conversion has already been made, don't do anything
        return
    
    # Some images are unlabelled and will be ignored
    image_files = {Path(f).stem for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))}

    file_base = 0
    iterations = 0

    for label in os.listdir(label_dir)[1:]:
        if label.endswith('.txt'):
            image_name = Path(label).stem

            if image_name in image_files:
                image_path = os.path.join(image_dir, image_name)
                label_path = os.path.join(label_dir, label)
                # Saves the broken down segments to the destination directory - no need to return
                file_base = breakdown(image_path, label_path, dst_dir, file_base)

        if iterations < 10:
            print(file_base, end=" ")  # To make sure that it is progressing - a progress bar of sorts
            iterations += 1
        else:
            print(file_base, end= '\r',)  # This doesn't work on the VS code terminal unfortunately 
            iterations = 0

def breakdown(image_path, label, dst_dir, file_base, segment=segments, targ_size = None):
    """
    Takes a single image and breaks it down into the target number of segments/ sub-images, saving each
    image in png format, and each binary label in a similarly named text file.

    Args:
        image path (string): The file path to the image being processed
        label (string): The file path to the label annotation
        file_base (int): The current count of the file naming system
        segments (int): The number of images that should be produced
        targ_size (int array): The dimensions of each sub-section produced, overwrites segments

    Returns:
        filebase (int): The current count of the file naming system.
    
    """
    with Image.open(image_path + ".jpg") as image:
        img_size, take_size = image.size[0], None

        # Can segment images by quantity (how many images you want) or by quantity (how large do you want the images)
        if targ_size is None:
            take_size = math.floor(img_size / segment)

        else:
            segment = math.floor(img_size / targ_size)
            take_size = img_size / segment

        # Particular to the Zebra label format
        label_data = []
        with open(label, 'r') as label_file:
            for line in label_file.readlines():
                if line[0] == '0':
                    parsed = list(map(float, line.split()[1:]))
                    entity_box = np.array([(parsed[i], parsed[i + 1]) for i in range(0, len(parsed), 2)])
                    label_data.append(entity_box * img_size)


        img = np.array(image)
        for i in range(segment):
            for j in range(segment):
                box_coordinates = [i * take_size, j * take_size, min((i + 1) * take_size, len(img[0])), min((j + 1) * take_size, len(img))]
                new_img = img[box_coordinates[1]: box_coordinates[3], box_coordinates[0]: box_coordinates[2]]
                new_image = Image.fromarray(new_img)

                crosswalk_intersection = False
                for crosswalk_box in label_data:
                    if check_box_intersection(box_coordinates, crosswalk_box):
                        # 1 means an image that contains a crosswalk (a significant portion of it)
                        crosswalk_intersection = True
                    else:
                        # 0 means background image (does not contain any significant portion of a crosswalk)
                        pass
                
                if crosswalk_intersection: 
                    with open(os.path.join(dst_dir, str(file_base)) + ".txt", 'w') as new_label_file:
                        new_label_file.write("1")
                        new_image.save(str(os.path.join(dst_dir, str(file_base))) + ".png")

                # As the crosswalks are sparse - this improves the balance of positive to negative cases for training
                else:
                    if random.randint(0, 4) >= 3:
                        with open(os.path.join(dst_dir, str(file_base)) + ".txt", 'w') as new_label_file:
                            new_label_file.write("0")
                            new_image.save(str(os.path.join(dst_dir, str(file_base))) + ".png")
                    
                # This ensures we don't overwrite previous segment files
                file_base += 1

    return file_base

# Only to be run if you run this file directly, else import and recall according to your requirements.
if __name__ == "__main__":
    convert_database_to_segments("zebra_annotations/zebra_images", "zebra_annotations/txt_annotations", "zebra_annotations/classification_data")