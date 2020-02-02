from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.logging import TestTubeLogger

from ..models import MODELS
from ..utils import make_directory, set_random_seed
from ..utils.metrics import EVAL


def train(arch, loader, args, test=True, return_test=False):
    set_random_seed(2434)
    device = list(range(args.device))
    save_dir = make_directory(args.logdir)

    model = MODELS[args.task](arch, loader, args, save_dir)

    exp = TestTubeLogger(save_dir=save_dir)
    exp.log_hyperparams(args)

    early_stop = EarlyStopping(
        monitor='avg_val_loss',
        patience=args.stop_num,
        verbose=False,
        mode='min')

    checkpoint = ModelCheckpoint(
        filepath=save_dir / "checkpoint",
        save_best_only=True,
        verbose=False,
        monitor='avg_val_loss',
        mode='min')

    backend = None if len(device) == 1 else "dp"

    trainer = Trainer(
        logger=exp,
        max_nb_epochs=args.epoch,
        checkpoint_callback=checkpoint,
        early_stop_callback=early_stop,
        gpus=device,
        distributed_backend=backend)

    trainer.fit(model)
    print("##### training finish #####")

    if test:
        model.load_best()
        trainer.test(model)
        EVAL[args.task](loader.get_label("test"), model.test_predict)
        print("##### test finish #####")

        if return_test:
            return model.test_predict
