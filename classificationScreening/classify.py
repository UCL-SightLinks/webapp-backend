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

vgg16_state_path = "models/VGG16_Full_State_Dict.pth"
# A prototype v2 version that's only been trained on 2000 images by transfer learning
mobileNet_path = "models/MobileNetV3_state_dict_big_train.pth"
data_path = "classificationScreening/classification_data"

classify = None
transform = None

def load_vgg_classifier(state_dict_path):
    # Ignore depreciation warnings --> It works fine for our needs
    model = models.vgg16()

    # Modifies fully connected layer to output binary class predictions
    model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 2)
    state_dict = torch.load(state_dict_path)
    model.load_state_dict(state_dict)

    model.eval()

    return model

# Only loads the classifier weights, in the case where it is transfer learning on only the top.
# This saves a significant amount of space
def partial_vgg_load(classifier_state_dict_path):
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

    model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 2)
    model.classifier.load_state_dict(classifier_state_dict_path)

    model.eval()

    return model

def load_resnet_classifier(state_dict_path):
    # Ignore depreciation warnings --> It works fine for our needs
    resnet = models.resnet18(pretrained=True)
    resnet.fc = torch.nn.Linear(resnet.fc.in_features, 1)

    state_dict = torch.load(state_dict_path)
    resnet.load_state_dict(state_dict)
    
    resnet.eval()
    return resnet

def load_mobileNet_classifier(state_dict_path):
    # Ignore depreciation warnings --> It works fine for our needs
    model = models.mobilenet_v3_small()
    model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 2)

    state_dict = torch.load(state_dict_path)
    model.load_state_dict(state_dict)

    model.eval()
    return model

# classify = load_vgg_classifier(vgg16_state_path)
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

    logit_pred = infer_model(image)

    probs = 1 / (1 + np.exp(-logit_pred.detach().numpy()))
    # prob = max(0, min(np.exp(logit_pred.detach().numpy())[0], 1))
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
        # label = label[0]  # Do not need to train so doesn't have to be a tensor
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
