from typing import Any

import lightning as pl
import torch
import torch.nn as nn
from torchmetrics import Accuracy, F1Score, Precision, Recall


class Learner(pl.LightningModule):
    """PyTorch Lightning wrapper for binary MIL classification (PD vs HC)."""

    def __init__(self,
                 model,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-4,
                 *args: Any,
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.model = model
        self.loss = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        metric_kwargs = dict(task="binary")
        self.train_acc = Accuracy(**metric_kwargs)
        self.val_acc = Accuracy(**metric_kwargs)
        self.val_f1 = F1Score(**metric_kwargs)
        self.val_precision = Precision(**metric_kwargs)
        self.val_recall = Recall(**metric_kwargs)

    def _step(self, batch):
        data, labels = batch
        logits = self.model(data)          # [B, 2]
        labels = labels.view(-1)
        loss = self.loss(logits, labels)
        preds = logits.argmax(dim=1)
        return loss, preds, labels

    def training_step(self, batch, batch_idx=None):
        loss, preds, labels = self._step(batch)
        self.train_acc(preds, labels)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx=None):
        with torch.no_grad():
            loss, preds, labels = self._step(batch)
            self.val_acc(preds, labels)
            self.val_f1(preds, labels)
            self.val_precision(preds, labels)
            self.val_recall(preds, labels)
            self.log("validation_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log("val_f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log("val_precision", self.val_precision, on_step=False, on_epoch=True, logger=True)
            self.log("val_recall", self.val_recall, on_step=False, on_epoch=True, logger=True)
        return {"validation_loss": loss}

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def forward(self, x):
        return self.model(x)
