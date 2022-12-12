import pytorch_lightning as pl
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from cai.dataset.dataset import HPASSDataset
from cai.utils.parser import _parse_transforms


class HPASSCDataModule(pl.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.catalog = kwargs.get("catalog")
        self.dataset_cfg = kwargs.get("dataset")
        self.dataloader_cfg = kwargs.get("dataloader")
        #self.is_prepared = False

    def setup(self, **kwargs):
        self.prepare_data()

    def prepare_data(self, **kwargs):
        self.dataset = HPASSDataset(**self.dataset_cfg)
        self.train_dataset, self.val_dataset = random_split(self.dataset, [0.8, 0.2])

        # should init test dataset here

    def train_dataloader(self):
        #self.phase = "train"
        return DataLoader(self.train_dataset, **self.dataloader_cfg)

    def val_dataloader(self):
        #self.phase = "val"
        return DataLoader(self.val_dataset, **self.dataloader_cfg)

    def test_dataloader(self):
        # separate test data
        # TODO
        #self.phase = "test"
        return None

if __name__ == "__main__":
    pass
