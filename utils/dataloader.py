from torch.utils.data import DataLoader
from .set_seed import worker_init_fn
from torchvision.datasets import ImageFolder


class MyDataLoader:
    def __init__(self, batch, num_worker):
        self.batch = batch
        self.num_worker = num_worker
        self.train = None
        self.valid = None
        self.test = None

    def set_train(self, train):
        self.train = train

    def set_valid(self, valid):
        self.valid = valid

    def set_test(self, test):
        self.test = test

    def get_label(self, data_name="test"):
        data = getattr(self, data_name)
        if isinstance(data, ImageFolder):
            return data.targets
        elif hasattr(data, "y"):
            return data.y

    def get_data(self, data_name="test"):
        data = getattr(self, data_name)
        if isinstance(data, ImageFolder):
            X, _ = zip(*data.imgs)
            return X
        elif hasattr(data, "X"):
            return data.X

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
        if self.test is None:
            return None
        return DataLoader(self.test,
                          batch_size=self.batch,
                          shuffle=False,
                          num_workers=self.num_worker,
                          worker_init_fn=worker_init_fn,
                          pin_memory=True)
