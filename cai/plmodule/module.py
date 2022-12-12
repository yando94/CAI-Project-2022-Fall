import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import transformers.models.vit as vit_models

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

        
        self.model = vit_models.ViTForImageClassification(
            vit_models.ViTConfig.from_pretrained(**vit_config) if pretrained else
            vit_models.ViTConfig(**vit_config)
        )

        self.reduce_channel = nn.Conv2d(in_channels, 
                                        out_channels,
                                        1,
                                        1,
                                        0) if reduce_channel else nn.Identity()
        
        self.loss_fn = getattr(nn, loss_fn_cfg.get("name"))(**loss_fn_cfg.get("kwargs"))

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
        output = F.sigmoid(output.logits)
        loss = self.loss_fn(output, label)

        results = {
            "loss" : loss
        }
        return results

    def _log(self, *args, **kwargs):
        pass

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        output = self._step(batch, batch_idx, optimizer_idx, caller=None)
        self._log(batch, batch_idx, output, optimizer_idx, caller=None)

        return output

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        output = self._step(batch, batch_idx, optimizer_idx, caller=None)
        self._log(batch, batch_idx, output, optimizer_idx, caller=None)

        return output

    def test_step(self, batch, batch_idx, optimizer_idx=0):
        output = self._step(batch, batch_idx, optimizer_idx, caller=None)
        self._log(batch, batch_idx, output, optimizer_idx, caller=None)

        return output
