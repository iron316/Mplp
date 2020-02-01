from pathlib import Path

import pytorch_lightning as pl
import torch


class BaseModule(pl.LightningModule):
    def __init__(self, model, logdir):
        super(BaseModule, self).__init__()
        self.model = model
        self.logdir = Path(logdir)

    def get_model(self):
        self.load_best()
        return self.model

    def load_best(self):
        cp = torch.load(list((self.logdir / "checkpoint").glob("*.ckpt"))[0])
        self.load_state_dict(cp["state_dict"])
