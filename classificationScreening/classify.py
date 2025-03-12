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
    # Ignore depreciation warnings --> It works fine for our needs
    model = models.mobilenet_v3_small()
    model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(state_dict_path, map_location=device)
    model.load_state_dict(state_dict)

    model.eval()
    return model

classify = load_mobileNet_classifier(mobileNet_path)
transform = classUtils.vgg_transform

# Expects a numpy array image
def infer(image, infer_model=classify, infer_transform=transform):
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
    tensor_im = torchvision.transforms.functional.pil_to_tensor(image).float()/ 255
    prediction = infer(tensor_im)
    classification = prediction[0][0] > threshold
    return classification

# Expects a numpy image
def infer_and_display(image, threshold, actual_label, onlyWrong=False):
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
