# All the utilities required for the inference and further tuning of the classification models

import torch
from torchvision import transforms
from torch.utils.data import Dataset

import os
import numpy as np
from PIL import Image


# The custom dataset object I used to load the segmented image dataset
class CrosswalkDataset(Dataset):
    """
    A custom dataset class that is used to store and load the segmented image dataset used in training the 
    classifiaction model.
    """
    def __init__(self, src_dir, transform=None):
        """
        Args:
            src_dir (string): The path to the source directory for the data stored by our model
            transform (pytorch transform object): The pre-processing transform used before accessing any item in the dataset
        """
        self.src_dir = src_dir
        self.transform = transform

        dir_files = sorted(os.listdir(src_dir))
        self.image_paths = [file_path for file_path in dir_files if file_path.endswith((".png", ".jpg", ".jpeg"))]
        self.label_paths = [file_path for file_path in dir_files if file_path.endswith(".txt")]

    def __len__(self):
        """
        Returns the length of the dataset, required for the instanciation of the object
        """
        return len(self.image_paths)
    
    def __getitem__(self, index):
        """
        The access mechanism for any item in the dataset, loading an image from the source 
        directory along with its associated label. Each individual access loads the image again.

        Args:
            index (int): The index of the image to be retrieved in alphanumerical ordering by name of file
        
        Returns:
            (image, label):
                image (depends on transform): A transformed version of the image stored in memory.
                label (tensor): A binary class label for the retrieved image.
        """
        image_path = os.path.join(self.src_dir, self.image_paths[index])
        label_path = os.path.join(self.src_dir, self.label_paths[index])

        label = [0, 0]
        try:
            if np.array([int(open(label_path).read().strip())]) == 1:
                label = [1, 0]
            else:
                label = [0, 1]
        except:
            pass
        image =  Image.open(image_path)
        
        if self.transform is None:
            self.transform = transforms.ToTensor()
        return (self.transform(image), torch.FloatTensor(label))


# Mean and Std. have been chosen because they fit the data we're working with. For re-use, please adjust.
# The image does not have to be resized within the transform, the global pooling layer should deal with this
# by the model (theoretically, we have not had to worry about this occuring yet), but resizing prevent
# potential errors if the models chosen do not do this.
vgg_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],  std=[0.3, 0.3, 0.3])
])

res_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],  std=[0.3, 0.3, 0.3])
])

