import os
import pandas as pd
import numpy as np
import torch
import torch.utils
from torch.utils.data import Dataset, DataLoader
from skimage import io
from torchvision import transforms


class CatsAndDogs(Dataset):
    def __init__(self, annotations_file, root_dir, transforms = None):
        super(CatsAndDogs, self).__init__()
        self.annotations = pd.read_csv(annotations_file)
        self.root_dir = root_dir
        self.transforms = transforms

    def __len__(self):
        return self.annotations.shape[0]

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(img_path)

        label = int(self.annotations.iloc[index, 1])
        label = torch.tensor(label)

        if self.transforms:
            image = self.transforms(image)

        return (image, label)
    

dataset = CatsAndDogs("./dataset/cats_dogs.csv", "./dataset/cats_dogs_resized/", transforms = transforms.ToTensor())

size = int(len(dataset))
train_set, test_set = torch.utils.data.random_split(dataset, [int(0.8*size), size - int(0.8*size)])

train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
test_loader = DataLoader(test_set, batch_size=4, shuffle=True)

for image, label in train_loader:
    image = image.permute(0, 3, 1, 2)
    print(image, label)