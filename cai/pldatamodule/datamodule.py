import pytorch_lightning as pl
import pandas as pd


class HPASSDataModule(pl.LightningDataModule):
    def __init__(self, catalog: str):
        self.dataset = HPASSDataset(catalog)


    