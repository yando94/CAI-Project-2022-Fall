import logging
import torch
import os
import pytorch_lightning as pl
import logging
from omegaconf import DictConfig, OmegaConf

if __name__ == "__main__":
    import sys
    sys.path.insert(0,"/home/yando/Workspace/CAI")

from cai.utils.loader import _load_pldatamodule, _load_plmodule
from cai.plmodule.module import HPASSCModule
from cai.pldatamodule.datamodule import HPASSCDataModule

class Application:
    def __init__(self, cfg):
        print(cfg)
        
        # load configs: Trainer, Module, DataModule
        self.trainer_cfg = OmegaConf.to_container(cfg.trainer)
        self.module_cfg = OmegaConf.to_container(cfg.module)
        self.datamodule_cfg = OmegaConf.to_container(cfg.datamodule)
        
        # load run configs used in self.run
        self.run_cfg = cfg.run

        # create logger
        self.logger = logging.getLogger(__name__)

    def run(self):
        # initialize required objects
        self.trainer = pl.Trainer(**self.trainer_cfg)
        #self.module = _load_plmodule(**self.module_cfg)
        self.module = HPASSCModule(**self.module_cfg)
        #self.datamodule = _load_pldatamodule(**self.datamodule_cfg)
        self.datamodule = HPASSCDataModule(**self.datamodule_cfg)

        # datamodule setup
        self.datamodule.setup()

        # run
        result = Application._run[self.run_cfg.type](self, **self.run_cfg.kwargs)
        return result

    def _train_run(self, **kwargs):
        return self.trainer.fit(self.module, datamodule=self.datamodule)

    def _test_run(self, ckpt=None, **kwargs):
        if ckpt is None:
            self.logger.warning("Running test with no ckpt!")
        result = self.trainer.test(self.module, 
                                   datamodule=self.datamodule,
                                   ckpt_path=ckpt)
        return result

    _run = dict(train = _train_run,
                test = _test_run)

if __name__ == "__main__":

    cfg = OmegaConf.create({'module': {'in_channels': 4, 'out_channels': 3, "reduce_channel": True, 'pretrained': True, 'vit_config': {'pretrained_model_name_or_path': 'google/vit-base-patch16-224', 'num_labels': 19}, 'configs': {'optimizer': {'module': 'torch', 'name': 'SGD', 'kwargs': {'lr': 0.001, 'momentum': 0.9}}, 'scheduler': {'module': 'torch', 'name': 'CosineAnnealingLR', 'kwargs': {'T_max': 100, 'eta_min': 1e-07}}, 'kld_weight': 0.1}, 'loss_fn': {'name': 'MultiLabelSoftMarginLoss', 'kwargs': {'weight': None, 'size_average': None, 'reduce': None, 'reduction': 'mean'}}}, 'datamodule': {'dataset': {'base_dir': '/mnt/yando/Users/yando/hpa-single-cell-image-classification', 'phase': 'train', 'catalog': '/home/yando/Workspace/CAI/catalog/train.csv', 'transforms': [{'module': 'tv_transforms', 'name': 'RandomRotation', 'kwargs': {'degrees': 180.0, 'expand': False}}, {'module': 'tv_transforms', 'name': 'CenterCrop', 'kwargs': {'size': [1500, 1500]}}, {'module': 'tv_transforms', 'name': 'RandomResizedCrop', 'kwargs': {'size': [224, 224], 'scale': [0.25, 1.0], 'ratio': [1.0, 1.0], 'antialias': True}}]}, 'dataloader': {'batch_size': 20, 'shuffle': False, 'sampler': None, 'batch_sampler': None, 'num_workers': 10, 'collate_fn': None, 'pin_memory': False, 'drop_last': False, 'timeout': 0, 'worker_init_fn': None, 'prefetch_factor': 2, 'persistent_workers': False}}, 'trainer': {'accelerator': 'gpu', 'accumulate_grad_batches': 1, 'amp_backend': 'native', 'auto_scale_batch_size': None, 'auto_select_gpus': False, 'auto_lr_find': False, 'benchmark': False, 'deterministic': False, 'callbacks': None, 'check_val_every_n_epoch': 10, 'devices': [0], 'enable_checkpointing': True, 'fast_dev_run': False, 'gradient_clip_val': 20.0, 'limit_train_batches': 1.0, 'log_every_n_steps': 6, 'max_epochs': 5000, 'min_epochs': 1, 'max_steps': -1, 'min_steps': None, 'max_time': None, 'precision': 32}, 'run': {'type': 'train', 'kwargs': {}}, 'base_dir': '/home/yando/NAS01/Users/yando/Experiments/cai/'})
    app = Application(cfg)
    app.run()