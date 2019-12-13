from torch.utils.data import DataLoader
from utils.set_seed import worker_init_fn


class MyDataLoader:
    def __init__(self, batch, num_worker):
        self.batch = batch
        self.num_worker = num_worker

    def set_train(self):
        raise NotImplementedError

    def set_valid(self):
        raise NotImplementedError

    def set_test(self):
        raise NotImplementedError

    @property
    def train_loader(self):
        return DataLoader(self.train,
                          batch_size=self.batch,
                          shuffle=True,
                          num_workers=self.num_worker,
                          worker_init_fn=worker_init_fn,
                          pin_memory=True)

    @property
    def valid_loader(self):
        return DataLoader(self.valid,
                          batch_size=self.batch,
                          shuffle=False,
                          num_workers=self.num_worker,
                          worker_init_fn=worker_init_fn,
                          pin_memory=True)

    @property
    def test_loader(self):
        return DataLoader(self.test,
                          batch_size=self.batch,
                          shuffle=False,
                          num_workers=self.num_worker,
                          worker_init_fn=worker_init_fn,
                          pin_memory=True)
