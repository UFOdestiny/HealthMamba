import os
import time

import numpy as np
import torch
import torch.nn as nn
from engine.metrics import Metrics


class HealthMambaEngine:
    def __init__(self, device, model, dataloader, scaler, optimizer, scheduler,
                 clip_grad_value, max_epochs, patience, log_dir, logger, args,
                 alpha=0.1, mc_samples=2, normalize=True, metric_list=None):
        self._device = device
        self.model = model.to(device)
        self._dataloader = dataloader
        self._scaler = scaler
        self._optimizer = optimizer
        self._lr_scheduler = scheduler
        self._clip_grad_value = clip_grad_value
        self._max_epochs = max_epochs
        self._patience = patience
        self._save_path = log_dir
        self._logger = logger
        self.args = args
        self._normalize = normalize
        self.alpha = alpha
        self.mc_samples = mc_samples
        self._iter_cnt = 0
        self._mask_value = torch.tensor(float("nan"))
        self.cqr_margin = 0.0

        if metric_list is None:
            metric_list = ["MAE", "MAPE", "RMSE", "MPIW", "COV"]
        self.metric = Metrics(metric_list, model.horizon)

        self._time_model = "{}_{}.pt".format(
            args.model_name, time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        )
        self._logger.info(f"{'Parameters':20s}: {self.model.param_num()}")
        self._logger.info(f"Model Save Path: {os.path.join(self._save_path, self._time_model)}")

    def _to_device(self, tensors):
        if isinstance(tensors, list):
            return [t.to(self._device) for t in tensors]
        return tensors.to(self._device)

    def _to_tensor(self, arrays):
        if isinstance(arrays, list):
            return [torch.tensor(a, dtype=torch.float32) for a in arrays]
        return torch.tensor(arrays, dtype=torch.float32)

    def _prepare_batch(self, batch):
        return self._to_device(self._to_tensor(batch))

    def _inverse_transform(self, tensors):
        def inv(t):
            return self._scaler.inverse_transform(t, device=self._device.type)
        if isinstance(tensors, list):
            return [inv(t) for t in tensors]
        return inv(tensors)

    def _compute_losses(self, mu, log_var, lower, upper, target):
        sigma = torch.exp(0.5 * log_var).clamp(min=1e-6)
        q_lo = self.alpha / 2
        q_hi = 1 - self.alpha / 2
        e_lo = target - lower
        e_hi = target - upper
        L_quant = (
            torch.where(e_lo >= 0, q_lo * e_lo, (q_lo - 1) * e_lo).mean()
            + torch.where(e_hi >= 0, q_hi * e_hi, (q_hi - 1) * e_hi).mean()
        )
        L_nll = (0.5 * log_var + 0.5 * (target - mu) ** 2 / (sigma ** 2 + 1e-6)).mean()
        r = (target - mu) / (sigma + 1e-6)
        L_calib = r.mean() ** 2 + (r.pow(2).mean() - 1) ** 2
        return L_quant + L_nll + L_calib

    def train_batch(self):
        self.model.train()
        self._dataloader["train_loader"].shuffle()
        mask_value = self._mask_value.to(self._device)

        for X, Y in self._dataloader["train_loader"].get_iterator():
            if self._iter_cnt == 0:
                self._logger.info(
                    f"Mask Value: {mask_value}\n\n" + "=" * 25 + "   Training   " + "=" * 25
                )
            self._optimizer.zero_grad()
            X, Y = self._prepare_batch([X, Y])

            mus, log_vars, lowers, uppers = [], [], [], []
            for _ in range(self.mc_samples):
                mu, lv, lo, up = self.model(X)
                mus.append(mu)
                log_vars.append(lv)
                lowers.append(lo)
                uppers.append(up)

            mu_stack = torch.stack(mus)
            mu_mean = mu_stack.mean(dim=0)
            L_param = ((mu_stack - mu_mean.detach()) ** 2).mean()

            mu_avg = mu_mean
            lv_avg = torch.stack(log_vars).mean(dim=0)
            lo_avg = torch.stack(lowers).mean(dim=0)
            up_avg = torch.stack(uppers).mean(dim=0)

            L_main = self._compute_losses(mu_avg, lv_avg, lo_avg, up_avg, Y)
            loss = L_main + L_param
            loss.backward()

            if self._clip_grad_value != 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self._clip_grad_value)
            self._optimizer.step()

            with torch.no_grad():
                mu_inv, lo_inv, up_inv, Y_inv = mu_avg, lo_avg, up_avg, Y
                if self._normalize:
                    mu_inv, lo_inv, up_inv, Y_inv = self._inverse_transform(
                        [mu_avg, lo_avg, up_avg, Y]
                    )
                self.metric.compute_batch(
                    mu_inv, Y_inv, mask_value, "train", lower=lo_inv, upper=up_inv
                )
            self._iter_cnt += 1

    def evaluate(self, mode, model_path=None, export=None, train_test=False):
        if mode == "test" and not train_test:
            if model_path:
                self.load_exact_model(model_path)
            else:
                self.load_model(self._save_path)

        self.model.eval()
        preds, lowers_all, uppers_all, labels = [], [], [], []

        with torch.no_grad():
            for X, Y in self._dataloader[mode + "_loader"].get_iterator():
                X, Y = self._prepare_batch([X, Y])
                mu, lv, lo, up = self.model(X)
                if self._normalize:
                    mu, lo, up, Y = self._inverse_transform([mu, lo, up, Y])

                if mode == "val":
                    self.metric.compute_batch(
                        mu, Y, self._mask_value.to(mu.device), "valid", lower=lo, upper=up
                    )
                else:
                    preds.append(mu.squeeze(-1).cpu())
                    lowers_all.append(lo.squeeze(-1).cpu())
                    uppers_all.append(up.squeeze(-1).cpu())
                    labels.append(Y.squeeze(-1).cpu())

        if mode == "val":
            return

        preds = torch.cat(preds, dim=0)
        lowers_all = torch.cat(lowers_all, dim=0)
        uppers_all = torch.cat(uppers_all, dim=0)
        labels = torch.cat(labels, dim=0)

        if mode in {"test", "export"}:
            mask_value = torch.tensor(float("nan"))
            for i in range(self.model.horizon):
                self.metric.compute_batch(
                    preds[:, i:i+1], labels[:, i:i+1], mask_value, "test",
                    lower=lowers_all[:, i:i+1], upper=uppers_all[:, i:i+1]
                )
            if not train_test:
                with self._logger.no_time():
                    self._logger.info("\n" + "=" * 25 + "     Test     " + "=" * 25)
                for msg in self.metric.get_test_msg():
                    self._logger.info(msg)
            if export:
                self.save_result(preds, lowers_all, uppers_all, labels)

    def train(self):
        wait = 0
        min_loss_val = np.inf
        for epoch in range(self._max_epochs):
            t1 = time.time()
            self.train_batch()
            t2 = time.time()
            v1 = time.time()
            self.evaluate("val")
            v2 = time.time()
            te1 = time.time()
            self.evaluate("test", train_test=True)
            te2 = time.time()

            valid_loss = self.metric.get_valid_loss()

            if self._lr_scheduler is None:
                cur_lr = self._optimizer.param_groups[0]["lr"]
            else:
                cur_lr = self._lr_scheduler.get_last_lr()[0]
                self._lr_scheduler.step()

            msg = self.metric.get_epoch_msg(epoch + 1, cur_lr, t2 - t1, v2 - v1, te2 - te1)
            self._logger.info(msg)

            if valid_loss < min_loss_val:
                if valid_loss == 0:
                    self._logger.info("Something went WRONG!")
                    break
                self.save_model(self._save_path)
                self._logger.info("Val loss: {:.3f} -> {:.3f}".format(min_loss_val, valid_loss))
                min_loss_val = valid_loss
                wait = 0
            else:
                wait += 1
                if wait == self._patience:
                    self._logger.info(
                        "Early stop at epoch {}, loss = {:.6f}".format(epoch + 1, min_loss_val)
                    )
                    break

        self.evaluate("test", export=self.args.export)

    def mc_inference(self, X, M=10):
        self.model.train()
        mus, sigmas, lowers, uppers = [], [], [], []
        with torch.no_grad():
            for _ in range(M):
                mu, lv, lo, up = self.model(X)
                mus.append(mu)
                sigmas.append(torch.exp(0.5 * lv))
                lowers.append(lo)
                uppers.append(up)
        mu_stack = torch.stack(mus)
        sigma_stack = torch.stack(sigmas)
        mu_mean = mu_stack.mean(dim=0)
        aleatoric = sigma_stack.pow(2).mean(dim=0)
        epistemic = (mu_stack - mu_mean).pow(2).mean(dim=0)
        total_var = aleatoric + epistemic
        lo_mean = torch.stack(lowers).mean(dim=0)
        up_mean = torch.stack(uppers).mean(dim=0)
        return mu_mean, total_var, lo_mean, up_mean

    def calibrate(self, alpha=None):
        if alpha is None:
            alpha = self.alpha
        self.model.eval()
        scores = []
        with torch.no_grad():
            for X, Y in self._dataloader["val_loader"].get_iterator():
                X, Y = self._prepare_batch([X, Y])
                mu, lv, lo, up = self.model(X)
                if self._normalize:
                    lo, up, Y = self._inverse_transform([lo, up, Y])
                score = torch.maximum(lo - Y, Y - up).clamp(min=0)
                scores.append(score.cpu())
        scores = torch.cat(scores, dim=0).reshape(-1)
        c = torch.quantile(scores, 1 - alpha).item()
        self.cqr_margin = c
        self._logger.info(f"CQR calibration margin: {c:.4f}")
        return c

    def save_model(self, save_path):
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(save_path, self._time_model))

    def load_model(self, save_path):
        f = os.path.join(save_path, self._time_model)
        if not os.path.exists(f):
            models = [i for i in os.listdir(save_path) if i.endswith(".pt")]
            if not models:
                self._logger.info(f"Model {f} Not Exist.")
                return
            models.sort(key=lambda fn: os.path.getmtime(os.path.join(save_path, fn)))
            f = os.path.join(save_path, models[-1])
            self._logger.info(f"Try the Newest Model {os.path.basename(f)}.")
        self.model.load_state_dict(torch.load(f, weights_only=False, map_location=self._device))

    def load_exact_model(self, path):
        self.model.load_state_dict(torch.load(path, weights_only=False, map_location=self._device))

    def save_result(self, preds, lowers, uppers, labels):
        result = torch.stack([lowers, preds, uppers, labels], dim=0)
        base_name = f"{self.args.model_name}-{self.args.dataset}-res"
        save_name = f"{base_name}.npy"
        path = os.path.join(self._save_path, save_name)
        suffix = 1
        while os.path.exists(path):
            save_name = f"{base_name}_{suffix}.npy"
            path = os.path.join(self._save_path, save_name)
            suffix += 1
        np.save(path, result.numpy())
        self._logger.info(f"Results Save Path: {path} | Shape: {result.shape}")
