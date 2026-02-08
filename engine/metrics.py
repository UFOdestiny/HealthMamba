import numpy as np
import torch
import torch.nn.functional as F


class Metrics:
    def __init__(self, metric_lst, horizon=1):
        self.dic = {
            "MAE": masked_mae, "RMSE": masked_rmse, "MAPE": masked_mape,
            "CRPS": masked_crps, "KL": masked_kl,
            "MPIW": masked_mpiw, "COV": masked_coverage,
            "WINK": masked_wink, "IS": masked_IS,
        }
        self.horizon = horizon
        self.metric_lst = [m for m in metric_lst if m in self.dic]
        self.metric_func = [self.dic[m] for m in self.metric_lst]
        self.N = len(self.metric_lst)
        self.train_res = [[] for _ in range(self.N)]
        self.valid_res = [[] for _ in range(self.N)]
        self.test_res = [[] for _ in range(self.N)]
        self._build_formatter()

    def _build_formatter(self):
        parts = ["Epoch: {:d}, "]
        for prefix in ["Tr", "V", "Te"]:
            parts.extend([f"{prefix} {m}: {{:.3f}}, " for m in self.metric_lst])
        parts.append("LR: {:.4e}, Tr: {:.1f}s, V: {:.1f}s, Te: {:.1f}s")
        self.epoch_fmt = "".join(parts)

    def compute_batch(self, preds, labels, null_val, mode, lower=None, upper=None):
        null_t = _align(null_val, preds)
        storage = {"train": self.train_res, "valid": self.valid_res, "test": self.test_res}[mode]
        for i, name in enumerate(self.metric_lst):
            func = self.metric_func[i]
            if name in {"MPIW"}:
                val = func(lower, upper) if lower is not None else torch.tensor(0.0)
            elif name in {"COV", "WINK", "IS"}:
                val = func(lower, upper, labels) if lower is not None else torch.tensor(0.0)
            else:
                val = func(preds, labels, null_t)
            storage[i].append(val.detach().item() if torch.is_tensor(val) else float(val))

    def get_valid_loss(self):
        return np.mean(self.valid_res[0]) if self.valid_res[0] else np.inf

    def get_test_loss(self):
        return np.mean(self.test_res[0]) if self.test_res[0] else np.inf

    def get_epoch_msg(self, epoch, lr, t_train, t_val, t_test):
        tr = [np.mean(r) if r else 0.0 for r in self.train_res]
        va = [np.mean(r) if r else 0.0 for r in self.valid_res]
        te = [np.mean(r) if r else 0.0 for r in self.test_res]
        msg = self.epoch_fmt.format(epoch, *tr, *va, *te, lr, t_train, t_val, t_test)
        self._reset()
        return msg

    def get_test_msg(self):
        fmt = ", ".join([f"{m}: {{:.3f}}" for m in self.metric_lst])
        msgs = []
        for i in range(self.horizon):
            vals = [r[i] for r in self.test_res]
            msgs.append(f"Horizon {i+1}: " + fmt.format(*vals))
        avg = [np.mean(r) for r in self.test_res]
        msgs.append("Average:    " + fmt.format(*avg))
        self._reset()
        return msgs

    def _reset(self):
        self.train_res = [[] for _ in range(self.N)]
        self.valid_res = [[] for _ in range(self.N)]
        self.test_res = [[] for _ in range(self.N)]


def _align(val, ref):
    if torch.is_tensor(val):
        return val.to(device=ref.device, dtype=ref.dtype)
    return torch.tensor(val, device=ref.device, dtype=ref.dtype)


def _get_mask(labels, null_val):
    null_val = _align(null_val, labels)
    if torch.isnan(null_val):
        return (~torch.isnan(labels)).float()
    return (labels != null_val).float()


def _masked_mean(loss, labels, null_val):
    mask = _get_mask(labels, null_val)
    count = mask.sum()
    if count == 0:
        return torch.tensor(0.0, device=loss.device)
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss * mask)
    return loss.sum() / count


def masked_mae(preds, labels, null_val):
    return _masked_mean(torch.abs(preds - labels), labels, null_val)


def masked_mse(preds, labels, null_val):
    return _masked_mean((preds - labels) ** 2, labels, null_val)


def masked_rmse(preds, labels, null_val):
    return torch.sqrt(masked_mse(preds, labels, null_val))


def masked_mape(preds, labels, null_val):
    loss = torch.abs(preds - labels) / torch.abs(labels).clamp(min=1e-5)
    return _masked_mean(loss, labels, labels.new_tensor(0.0)) * 100


def masked_kl(preds, labels, null_val):
    loss = labels * torch.log((labels + 1e-5) / (preds + 1e-5))
    return _masked_mean(loss, labels, null_val)


def masked_crps(preds, labels, null_val):
    try:
        import properscoring as ps
        p = preds.detach().cpu().numpy()
        l = labels.detach().cpu().numpy()
        return torch.tensor(ps.crps_ensemble(l, p).mean(), device=preds.device)
    except ImportError:
        return torch.tensor(0.0, device=preds.device)


def masked_mpiw(lower, upper, null_val=None):
    return (upper - lower).mean()


def masked_wink(lower, upper, labels, alpha=0.1):
    zero = torch.tensor(0.0, device=lower.device)
    score = upper - lower
    score = score + (2 / alpha) * torch.maximum(lower - labels, zero)
    score = score + (2 / alpha) * torch.maximum(labels - upper, zero)
    return score.mean()


def masked_coverage(lower, upper, labels, alpha=None):
    return ((labels >= lower) & (labels <= upper)).float().mean() * 100


def masked_IS(lower, upper, labels, alpha=0.1):
    width = upper - lower
    below = (labels < lower).float()
    above = (labels > upper).float()
    score = width + (2 / alpha) * (lower - labels) * below + (2 / alpha) * (labels - upper) * above
    return score.mean()
