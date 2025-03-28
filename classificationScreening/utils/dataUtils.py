from shapely.geometry import Polygon

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