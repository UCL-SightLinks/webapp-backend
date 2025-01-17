from shapely.geometry import Polygon

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