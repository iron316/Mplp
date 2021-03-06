import torch
import torch.nn as nn

import pytorch_lightning as pl
from .base import BaseModule


class RegressionModel(BaseModule):
    def __init__(self, model, loader, args, logdir):
        super(RegressionModel, self).__init__(
            model,
            logdir
        )
        self.hparams = args
        self.loader = loader
        self.loss_func = nn.MSELoss(reduction="none")
        self.test_predict

    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)]

    def forward(self, X):
        out = self.model(X)
        return out

    def training_step(self, batch, batch_nb):
        X, y = batch
        target = y.type(torch.float32).view(-1, 1)
        predict = self.forward(X)
        loss = self.loss_func(predict, target).mean().unsqueeze(0)
        logs = {}
        logs["train_loss"] = loss
        return {"loss": loss,
                "log": logs}

    def validation_step(self, batch, batch_nb):
        X, y = batch
        target = y.type(torch.float32).view(-1, 1)
        predict = self.forward(X)
        loss = self.loss_func(predict, target).mean().unsqueeze(0)
        return {"val_loss": loss}

    def test_step(self, batch, batch_nb):
        X, y = batch
        target = y.type(torch.float32).view(-1, 1)
        predict = self.forward(X)
        loss = self.loss_func(predict, target).mean().unsqueeze(0)
        return {"test_predict": predict,
                "test_loss": loss}

    def validation_end(self, outputs):
        avg_val_loss = 0.0
        for output in outputs:
            avg_val_loss += output["val_loss"].mean() / len(outputs)
        logs = {}
        logs["valid_loss"] = avg_val_loss
        return {"avg_val_loss": avg_val_loss,
                "progress_bar": logs,
                "log": logs}

    def test_end(self, outputs):
        avg_test_loss = 0.0
        self.test_predict = []
        for output in outputs:
            self.test_predict.extend(output["test_predict"].flatten().cpu().numpy().tolist())
            avg_test_loss += output["test_loss"].mean() / len(outputs)
        logs = {}
        logs["test_loss"] = avg_test_loss
        return {"avg_test_loss": avg_test_loss,
                "progress_bar": logs}

    @pl.data_loader
    def tng_dataloader(self):
        return self.loader.train_loader

    @pl.data_loader
    def val_dataloader(self):
        return self.loader.valid_loader

    @pl.data_loader
    def test_dataloader(self):
        return self.loader.test_loader
