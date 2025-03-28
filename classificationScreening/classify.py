import sys
import os
import torch
import torchvision
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
import random

import warnings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from classificationScreening.utils import classUtils

# Torchvision's models utils has a depreciation warning for the pretrained parameter in its instantiation but we don't use that
warnings.filterwarnings(
    action='ignore',
    category=DeprecationWarning,
    module=r'.*'
)

mobileNet_path = "models/mn3_vs55.pth"
data_path = "classificationScreening/classification_data"

classify = None
transform = None


def load_mobileNet_classifier(state_dict_path):
    """
    Initialises the weights of the Mobile Net v3 model architecture to the pre-trained weights
    stored in the model state dictionairy in the 'models' directory
    
    Args:
        state_dict_path (string): The path to the state dictionairy relative to the function call
    
    Returns:
        model (MobileNetV3 object): An initialised mobilenet v3 model with the saved weights,
          in evaluation mode so the weights will not be changed
    """
    model = models.mobilenet_v3_small()
    model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(state_dict_path, map_location=device)
    model.load_state_dict(state_dict)

    model.eval()
    return model

classify = load_mobileNet_classifier(mobileNet_path)
transform = classUtils.vgg_transform

def infer(image, infer_model=classify, infer_transform=transform):
    """
    Applies a binary classification model to an input image array, generalising the inference
    process to be model-independent. 

    Args:
        image (numpy array): The input image(s) in the form of a numpy array of form
            [channels: [width: [height:]]] or [batch: [channels: [width: [height:]]]]
        infer_model (pytorch model object): Any binary classification pytorch model object with a forward method
        infer_transform (pytorch transform object): Any pre-processing required for the model to run
    
    Returns:
        probs (tensor): A list of binary classifaction confidences in the range [0, 1], 
            where each member of the batch sums to 1.

    """
    # If infer model and transform have not been initialised
    if infer_model is None or infer_transform is None:
        raise TypeError("Error: The inference classes have not been initialised properly.")
    if not torch.is_tensor(image):
        image = infer_transform(image)
    
    # Expects batches - this adds another dimensions 
    if len(image.shape) <= 3:
        image = image.unsqueeze(0)

    
    pred = torch.sigmoid(infer_model(image))
    probs = pred.detach().numpy()

    return probs

def PIL_infer(image, threshold=0.35):
    """
    A wrapper function for the infer function that ensures compatibility with the PIL image library format
    and that works for a single image only. Uses the system default model.

    Args:
        image (PIL image): The image that will be classified in PIL format
        threshold (float): The confidence threshold for postiive classification
    
    Returns:
        classification (boolean): Whether the image is likely to be the target object.
    """
    tensor_im = torchvision.transforms.functional.pil_to_tensor(image).float()/ 255
    prediction = infer(tensor_im)
    classification = prediction[0][0] > threshold
    return classification

# Expects a numpy image
def infer_and_display(image, threshold, actual_label, onlyWrong=False):
    """
    A wrapper for the infer function that allows a visual representation of the model's classification, for 
    demonstration, debuggin and model fine-tuning. Uses the system default model.

    Args:
        image (numpy array): The batch of images that will be classified
        threshold (float): The minimum probability required for a positive classification
        actual label (boolean array): The ground truth class labels of the input image array
        onlyWrong (boolean): Whether to exclusively display incorrect classifications

    Returns:
        prediction (boolean array): The set of classifications made by the model, only returned if they were all correct
        probability (tensor array): The set of probabilities assigned to each class by the binary classification model
    """
    probability = infer(image)
    prediction = probability > threshold
    is_correct = (actual_label[0] == 1) == prediction

    if onlyWrong and is_correct:
        return prediction
    
    plt.imshow(torch.permute(image, (1, 2, 0)).detach().numpy())
    plt.title(f"Prediction: {prediction[0][0]} with confidence {probability[0][0]}%, Actual: {actual_label[0] == 1}")
    plt.axis("off")
    plt.show()

    return probability


def example_init(examples=20, display=True):
    """
    An example function for the classification process, for demonstration purposes and as a tutorial for usage

    Args:
        examples (int): The number of images to be loaded from the training dataset.
        display (boolean): Whether to display the results of each classification, which pauses the program until
            each individual classification display window is closed.

    Returns:
        None
    """
    dataset = classUtils.CrosswalkDataset(data_path)
    
    random_points = [random.randint(0, len(dataset)-1) for i in range(examples)]
    correct, incorrect, falsepos, falseneg = 0, 0, 0, 0
    for point in random_points:
        image, label = dataset[point]

        class_guess = [0, 1]
        if infer(image)[0][0] > 0.5:
            class_guess = [1, 0]
        if class_guess == label.tolist():
            correct += 1
        else:
            if class_guess[0]:
                falsepos += 1
            else:
                falseneg += 1
            incorrect += 1
        
        if display:
            print(f"Prediction of {infer_and_display(image, 0.4, label)}% of a crosswalk (Crosswalk: {label[0]==1})")
    print(f"correct: {correct}, incorrect: {incorrect}, of which false positives were {falsepos} and false negatives were {falseneg}")