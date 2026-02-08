import numpy as np
import torch


class LogScaler:
    def transform(self, data):
        return np.log(data + 1)

    def inverse_transform(self, data, device=None):
        if isinstance(data, np.ndarray):
            return np.exp(data) - 1
        return torch.exp(data) - 1


class LogMinMaxScaler:
    def __init__(self, data_min=0, data_max=0):
        self.data_min_ = torch.tensor(data_min, dtype=torch.float32)
        self.data_max_ = torch.tensor(data_max, dtype=torch.float32)

    def transform(self, data):
        log_data = np.log1p(data)
        mn = self.data_min_.numpy()
        mx = self.data_max_.numpy()
        return (log_data - mn) / (mx - mn)

    def inverse_transform(self, data, device=None):
        span = self.data_max_ - self.data_min_
        if isinstance(data, torch.Tensor):
            return torch.expm1(data * span + self.data_min_)
        return np.expm1(data * span.numpy() + self.data_min_.numpy())
