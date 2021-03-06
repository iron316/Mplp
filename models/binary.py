import torch
import torch.nn as nn

import pytorch_lightning as pl
from .base import BaseModule


class BinaryModel(BaseModule):
    def __init__(self, model, loader, args, logdir):
        super(BinaryModel, self).__init__(
            model,
            logdir
        )
        self.hparams = args
        self.loader = loader
        self.loss_func = nn.BCEWithLogitsLoss(reduction="none")
        self.test_predict = []

    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)]

    def forward(self, X):
        out = self.model(X)
        return out

    def _accuracy(self, output, y):
        predict = (output > 0).long().squeeze()
        return (predict == y).sum().unsqueeze(0).float() / output.size(0)

    def training_step(self, batch, batch_nb):
        X, y = batch
        target = y.type(torch.float32).view(-1, 1)
        predict = self.forward(X)
        loss = self.loss_func(predict, target).mean().unsqueeze(0)
        logs = {}
        logs["train_loss"] = loss
        logs["train_accuracy"] = self._accuracy(predict, y)
        return {"loss": loss,
                "progress_bar": logs,
                "log": logs}

    def validation_step(self, batch, batch_nb):
        X, y = batch
        target = y.type(torch.float32).view(-1, 1)
        predict = self.forward(X)
        loss = self.loss_func(predict, target).mean().unsqueeze(0)
        return {"val_loss": loss,
                "val_accuracy": self._accuracy(predict, y)}

    def test_step(self, batch, batch_nb):
        X, y = batch
        target = y.type(torch.float32).view(-1, 1)
        predict = self.forward(X)
        loss = self.loss_func(predict, target).mean().unsqueeze(0)
        return {"test_predict": predict,
                "test_loss": loss,
                "test_accuracy": self._accuracy(predict, y)}

    def validation_end(self, outputs):
        avg_val_accuracy = 0.0
        avg_val_loss = 0.0
        for output in outputs:
            avg_val_loss += output["val_loss"].mean() / len(outputs)
            avg_val_accuracy += output["val_accuracy"].mean() / len(outputs)
        logs = {}
        logs["valid_loss"] = avg_val_loss
        logs["valid_accuracy"] = avg_val_accuracy
        return {"avg_val_loss": avg_val_loss,
                "progress_bar": logs,
                "log": logs}

    def test_end(self, outputs):
        avg_test_accuracy = 0.0
        avg_test_loss = 0.0
        self.test_predict = []
        for output in outputs:
            self.test_predict.extend(output["test_predict"].flatten().cpu().numpy().tolist())
            avg_test_loss += output["test_loss"].mean() / len(outputs)
            avg_test_accuracy += output["test_accuracy"].mean() / len(outputs)
        logs = {}
        logs["test_loss"] = avg_test_loss
        logs["valid_accuracy"] = avg_test_accuracy
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
