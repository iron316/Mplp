import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", "-t", type=str,
                        choices=["binary", "multiclass", "regressoin"],
                        help="choice training task")
    parser.add_argument("--device", "-d", type=int, default=1,
                        help="GPU ID (negative value indicates CPU)")
    parser.add_argument("--batch", "-b", type=int, default=32,
                        help="Number of images in each mini-batch")
    parser.add_argument("--epoch", "-e", type=int, default=150,
                        help="Number of sweeps over the dataset to train")
    parser.add_argument("--lr", "-l", type=float, default=1e-4,
                        help="Number of learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Adam's weight decay")
    parser.add_argument("--stop_num", "-s", type=int, default=100,
                        help="Number of Early Stopping")
    parser.add_argument("--logdir", type=str, default=None,
                        help="default is year_month_day_hour")
    args = parser.parse_args()

    return args
