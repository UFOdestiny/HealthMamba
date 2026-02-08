import math
import numpy as np
from pathlib import Path
from utils.scaler import LogScaler, LogMinMaxScaler


class DataLoader:
    def __init__(self, data, idx, seq_len, horizon, bs, logger, name=None, droplast=False):
        self.data = np.asarray(data)
        self.idx = np.asarray(idx)
        self.size = len(self.idx)
        self.bs = bs
        self.droplast = droplast
        self.num_batch = self.size // bs if droplast else math.ceil(self.size / bs)
        self.current_ind = 0
        self.x_offsets = np.arange(-(seq_len - 1), 1, 1)
        self.y_offsets = np.arange(1, horizon + 1, 1)
        self.seq_len = seq_len
        self.horizon = horizon
        loader_name = name or "loader"
        logger.info(f"{loader_name:5s} num: {self.size},\tBatch num: {self.num_batch}")

    def shuffle(self):
        perm = np.random.permutation(self.size)
        self.idx = self.idx[perm]

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start = self.bs * self.current_ind
                end = min(self.size, self.bs * (self.current_ind + 1))
                idx = np.asarray(self.idx[start:end]).reshape(-1)
                if len(idx) == 0:
                    break
                if self.droplast and len(idx) < self.bs:
                    self.current_ind += 1
                    continue
                x = self.data[idx[:, None] + self.x_offsets, ...].astype(np.float32)
                y = self.data[idx[:, None] + self.y_offsets, ...].astype(np.float32)
                yield x, y
                self.current_ind += 1

        return _wrapper()


def load_dataset(data_path, args, logger):
    data_dir = Path(data_path) / args.years
    ptr = np.load(data_dir / "his.npz")
    logger.info(f"{'Data shape':20s}: {ptr['data'].shape}")
    X = ptr["data"]
    dataloader = {}
    for cat in ["train", "val", "test"]:
        idx = np.load(data_dir / f"idx_{cat}.npy")
        dataloader[f"{cat}_loader"] = DataLoader(
            X, idx, args.seq_len, args.horizon, args.bs, logger, cat
        )
    scaler = LogMinMaxScaler(ptr["min"], ptr["max"]) if "min" in ptr else LogScaler()
    return dataloader, scaler


def load_adj(path):
    return np.load(path)
