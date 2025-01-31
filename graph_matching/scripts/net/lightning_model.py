"""
Lightning model for training and evaluation.
"""

import torch
import pytorch_lightning as pl
from easydict import EasyDict
from net.models import GModels
from typing import Tuple


class LightningModel(pl.LightningModule):
    """
    A PyTorch Lightning module encapsulating the graph model, training, validation, and testing logic.
    """

    def __init__(self, config: EasyDict):
        """
        Initialize the LightningModel.
        Args:
            config (EasyDict): Configuration object.
        """
        super().__init__()
        self.save_hyperparameters(config)
        self.model = GModels(config=config)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.model(X)

    def step(self, batch: Tuple[torch.Tensor, torch.Tensor], stage: str):
        X, label = batch
        pred = self.model(X).squeeze()
        loss = torch.nn.CrossEntropyLoss()(pred, label)
        acc = (pred.argmax(dim=1) == label).float().mean()

        self.log(f"{stage}_loss", loss, batch_size=X.size(0))
        self.log(f"{stage}_acc", acc, batch_size=X.size(0))
        return loss

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _):
        return self.step(batch, "train")

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _):
        with torch.no_grad():
            return self.step(batch, "val")

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _):
        with torch.no_grad():
            return self.step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=self.hparams.lr_factor, patience=1)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "train_loss"}}

