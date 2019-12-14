from pathlib import Path
import datetime


def make_directory(name):
    result = Path("result")
    result.mkdir(exist_ok=True)
    if name is not None:
        dir_name = name
    else:
        now = datetime.datetime.now()
        dir_name = datetime.datetime.strftime(now, "%y_%m_%d_%H")
    log_dir = result / dir_name
    log_dir.mkdir(exist_ok=True)

    return log_dir
