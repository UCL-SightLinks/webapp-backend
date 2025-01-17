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

# Should eventually be moved to a settings class
zebra_labels = "zebra_annotations/txt_annotations"
zebra_images = "zebra_annotations/zebra_images"
save_dir = "zebra_annotations/classification_data"
segments = 4

# Each box has format (x_1, y_1, x_2, y_2) - this does mean this is an estimate.
# box_1 is the bounding box of the crosswalk, and box_2 is that of the tile.
def check_box_intersection(box_1, box_2, threshold=0.6):
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


# Takes in a entity database, labelled for bounding box regression, and breaks down the images into
# smaller images, with labels for classification training
def convert_database_to_segments(image_dir, label_dir, dst_dir, overwrite=False):
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

convert_database_to_segments("zebra_annotations/zebra_images", "zebra_annotations/txt_annotations", "zebra_annotations/classification_data")
