import os
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset

from cai.utils.parser import _parse_label, _parse_images, _parse_transforms

class HPASSDataset(Dataset):
    # var names
    _var_id = "ID"
    _var_label = "Label"
    _label_names = [
            "Nucleoplasm",
            "Nuclear membrane",
            "Nucleoli",
            "Nucleoli fibrillar center",
            "Nuclear speckles",
            "Nuclear bodies",
            "Endoplasmic reticulum",
            "Golgi apparatus",
            "Intermediate filaments",
            "Actin filaments",
            "Microtublules",
            "Mitotic spindle",
            "Centrosome",
            "Plasma membrane",
            "Mitochondria",
            "Aggresome",
            "Cytosol",
            "Vesicles and punctate cytosolic patterns",
            "Negative"
            ]
    _colors = [
            "red",
            "green",
            "blue",
            "yellow"
            ]
    _color_names = {
            "red" : "microtubles",
            "green" : "POI", # protein of interest
            "blue" : "nucleus",
            "yellow" : "endoplasmic reticulum"
            }

    def __init__(self, **kwargs):
        self.base_dir = kwargs.get("base_dir")
        self.catalog = kwargs.get("catalog")
        data = pd.read_csv(os.path.join(self.base_dir, self.catalog))
        self.image_id = data["ID"]
        self.label = data["Label"].apply(_parse_label)
        self.phase = kwargs.get("phase")
        self.transform = _parse_transforms(kwargs.get("transforms"))

        self.transform = self.transform if self.transform else nn.Identity()

    def __getitem__(self, idx):
        images = [Image.open(os.path.join(self.base_dir, self.phase, self.image_id[idx] + "_" + color + ".png")) for color in HPASSDataset._colors]
        label = self.label[idx]

        images = _parse_images(images) # returns Tensor
        #label = _parse_label(label) # returns Tensor

        images = self.transform(images)
        #label = self.label_transform(label)

        return images, label

    def __len__(self):
        return len(self.label)

class HPASSPublicDataset(Dataset):
    def __init__(self, cfg, catalog):
        pass

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass
