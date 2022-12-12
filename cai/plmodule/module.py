import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import transformers.models.vit as vit_models

import torchvision.transforms as tv_transforms
import torchmetrics


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from cai.utils.loss_fn import sigmoidF1Loss

class HPASSCModule(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()

        in_channels = kwargs.get("in_channels")
        out_channels = kwargs.get("out_channels")
        reduce_channel = kwargs.get("reduce_channel")
        vit_config = kwargs.get("vit_config")
        pretrained = kwargs.get("pretrained")
        loss_fn_cfg = kwargs.get("loss_fn")
        
        self.configs = kwargs.get("configs")
        self.log_metrics = kwargs.get("log_metrics")

        self.classes = list(vit_config["label2id"].keys())

        
        self.model = vit_models.ViTForImageClassification(
            vit_models.ViTConfig.from_pretrained(**vit_config) if pretrained else
            vit_models.ViTConfig(**vit_config)
        )

        self.reduce_channel = nn.Conv2d(in_channels, 
                                        out_channels,
                                        1,
                                        1,
                                        0) if reduce_channel else nn.Identity()
        
        self.loss_fn = getattr(globals()[loss_fn_cfg.get("module")], loss_fn_cfg.get("name"))(**loss_fn_cfg.get("kwargs"))

    def forward(self, batch, **kwargs):
        batch = self.reduce_channel(batch)
        return self.model(batch, **kwargs)

    def configure_optimizers(self):
        optimizers = []
        schedulers = []
        optimizer_class = getattr(globals()[self.configs["optimizer"]["module"]].optim, self.configs["optimizer"]["name"])
        optimizer = optimizer_class(params=self.model.parameters(), **self.configs["optimizer"]["kwargs"])
        optimizers.append(optimizer)

        scheduler_class = getattr(globals()[self.configs["scheduler"]["module"]].optim.lr_scheduler, self.configs["scheduler"]["name"])
        scheduler = scheduler_class(optimizer = optimizer, **self.configs["scheduler"]["kwargs"])
        schedulers.append(scheduler)

        return optimizers if len(schedulers) < 1 else optimizers, schedulers


    def _step(self, batch, batch_idx, optimizer_idx = 0, caller = ""):
        image, label = batch
        output = self.forward(image)
        pred = torch.sigmoid(output.logits)
        loss = self.loss_fn(pred, label)

        results = {
            "loss" : loss,
            "logits" : output.logits,
            "label" : label
        }
        return results

    def _log(self, *args, **kwargs):
        results = args[0]
        caller = kwargs.get("caller")
        for log_metric in self.log_metrics:
            if log_metric["tphase"] == kwargs.get("tphase"):
                value = getattr(torchmetrics.functional, log_metric["name"])(results["logits"], results["label"], **log_metric["kwargs"])
                self.log(log_metric["name"], {caller: value})

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        output = self._step(batch, batch_idx, optimizer_idx, caller=None)
        self.log("loss", {"training" : output["loss"]})
        self._log(output, tphase="step", caller="training")

        return output

    def training_epoch_end(self, outputs):
        output = {}
        output["logits"] = torch.cat([o["logits"] for o in outputs])
        output["label"] = torch.cat([o["label"] for o in outputs])

        self._log(output, tphase="epoch", caller="training")

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        output = self._step(batch, batch_idx, optimizer_idx, caller=None)
        self.log("loss", {"validation": output["loss"]}, on_epoch=True)
        self._log(output, tphase="step", caller="validation")

        return output

    def validation_epoch_end(self, outputs):
        output = {}
        output["logits"] = torch.cat([o["logits"] for o in outputs])
        output["label"] = torch.cat([o["label"] for o in outputs])

        self._log(output, tphase="epoch", caller="validation")

    def test_step(self, batch, batch_idx, optimizer_idx=0):
        output = self._step(batch, batch_idx, optimizer_idx, caller=None)
        self._log(output, tphase="step")

        return output

    def test_epoch_end(self, outputs):
        output = {}
        output["logits"] = torch.cat([logits for logits in outputs['logits']])
        output["label"] = torch.cat(labels for labels in outputs["label"])
        
        self._log(outputs, tphase="epoch")
