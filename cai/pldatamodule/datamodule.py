import pytorch_lightning as pl
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from PIL import Image
from cai.dataset.dataset import HPASSDataset
from cai.utils.parser import _parse_transforms, _parse_label


class HPASSCDataModule(pl.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.dataset_cfg = kwargs.get("dataset")
        self.dataloader_cfg = kwargs.get("dataloader")
        self.use_wr_sampler = kwargs.get("use_wr_sampler")
        self.num_samples = kwargs.get("num_samples")
        #self.is_prepared = False

    def setup(self, **kwargs):
        self.prepare_data()

    def prepare_data(self, **kwargs):
        self.dataset = HPASSDataset(**self.dataset_cfg)
        self.train_dataset, self.val_dataset = random_split(self.dataset, [0.8, 0.2], torch.Generator().manual_seed(42))

        if self.use_wr_sampler:
            self.dataset.data["LabelTensor"] = self.dataset.data["Label"].apply(_parse_label)
            sample_weights = torch.tensor(1.0 / self.dataset.data["Label"].value_counts()[[str(i) for i in range(19)]])
            sample_weights = self.dataset.data["LabelTensor"].apply(lambda x : (sample_weights[x.nonzero()]).prod()).array
            self.sample_weights = sample_weights

        # should init test dataset here

    def train_dataloader(self):
        #self.phase = "train"
        self.train_sampler = WeightedRandomSampler(self.sample_weights[self.train_dataset.indices], self.dataloader_cfg["batch_size"] * self.num_samples, replacement=False) if self.use_wr_sampler else None
        return DataLoader(self.train_dataset, sampler = self.train_sampler, **self.dataloader_cfg)

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
