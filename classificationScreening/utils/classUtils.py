# All the utilities required for the inference and further tuning of the classification models

import torch
from torchvision import transforms
from torch.utils.data import Dataset

import os
import numpy as np
from PIL import Image


# The custom dataset object I used to load the segmented image dataset
class CrosswalkDataset(Dataset):
    def __init__(self, src_dir, transform=None):
        self.src_dir = src_dir
        self.transform = transform

        dir_files = sorted(os.listdir(src_dir))
        self.image_paths = [file_path for file_path in dir_files if file_path.endswith((".png", ".jpg", ".jpeg"))]
        self.label_paths = [file_path for file_path in dir_files if file_path.endswith(".txt")]

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
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


# Mean and Std. are chosen arbitrarily - need to be tuned
# The image do not have to be resized - the global pooling layer should technically deal with this, but I haven't tested this,
# so resizing prevents potential inaccuracies from occuring,
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


