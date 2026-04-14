import numpy as np
import torch
from torch import nn, optim

device = "cuda" if torch.cuda.is_available() else "cpu"

_SMOOTHNESS_A_PAIR = 5.0
_SMOOTHNESS_MU_PAIR = 1.0
_SMOOTHNESS_A_MULTI = 5.0
_SMOOTHNESS_MU_MULTI = 1.0


class PredicateInduction:
    def __init__(self, exponent=3, weight_decay_a=0.01, momentum=0.9,
                 nesterov=True, smoothness_scale=1.0) -> None:
        self.exponent = exponent
        self.weight_decay_a = weight_decay_a
        self.momentum = momentum
        self.nesterov = nesterov and momentum > 0
        self.smoothness_scale = smoothness_scale
        self.visualizations = []

    def predict(self, x, a, mu):
        return 1 / (1 + ((a.abs() * (x - mu).abs()).pow(self.exponent)).sum(1))

    def compute_predicate_sequence(self, x0, selected, attribute_names=[], n_iter=1000,
                                   eps=1e-4, balanced=True, priors=None, init_noise=0.0,
                                   random_init=False):
        n_points, n_features = x0.shape

        vmin, vmax = x0.min(0), x0.max(0)
        x = torch.from_numpy(x0.astype(np.float32)).to(device)
        label = torch.from_numpy(selected).float().to(device)

        mean, scale = x.mean(0), x.std(0) + eps
        x = (x - mean) / scale

        selection_centroids = torch.stack([x[sel_t].mean(0) for sel_t in selected], 0)
        selection_std = torch.stack([x[sel_t].std(0, correction=0) for sel_t in selected], 0) + eps

        if random_init:
            a = torch.abs(torch.randn(selection_centroids.shape)) + 0.5
            mu = torch.randn(selection_centroids.shape) * 0.3
        else:
            a = (1 / selection_std).clone()
            mu = selection_centroids.clone()

        if priors is not None:
            attr_idx = {name: k for k, name in enumerate(attribute_names)}
            for t, prior in enumerate(priors):
                if prior:
                    for attr, (lo, hi) in prior.items():
                        if attr in attr_idx and hi > lo:
                            k = attr_idx[attr]
                            lo_n = (float(lo) - mean[k].item()) / scale[k].item()
                            hi_n = (float(hi) - mean[k].item()) / scale[k].item()
                            mu[t, k] = (lo_n + hi_n) / 2.0
                            a[t, k] = 2.0 * scale[k].item() / (hi - lo)

        if init_noise > 0.0:
            a = a * torch.exp(torch.randn_like(a) * init_noise)
            mu = mu + torch.randn_like(mu) * init_noise

        a.requires_grad_(True)
        mu.requires_grad_(True)

        bce_per_brush = self._create_bce_per_brush(selected, n_points, x, balanced=balanced)

        optimizer = optim.SGD(
            [{"params": mu, "weight_decay": 0},
             {"params": a, "weight_decay": self.weight_decay_a}],
            lr=1e-2,
            momentum=self.momentum,
            nesterov=self.nesterov,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(n_iter, 1))

        self._optimize(selected, x, a, mu, label, bce_per_brush, optimizer, n_iter, scheduler)

        a.detach_()
        mu.detach_()

        return self._generate_predicates(selected, x0, a, mu, mean, scale, vmin, vmax, attribute_names)

    def _create_bce_per_brush(self, selected, n_points, x, balanced=True):
        bce_per_brush = []
        for st in selected:
            n_selected = st.sum()
            n_unselected = n_points - n_selected
            instance_weight = torch.ones(x.shape[0]).to(device)
            if balanced:
                st_t = torch.from_numpy(st.astype(np.float32)).to(device)
                instance_weight *= st_t * (n_unselected / n_points) + (1 - st_t) * (n_selected / n_points)
            bce_per_brush.append(nn.BCELoss(weight=instance_weight))
        return bce_per_brush

    def _optimize(self, selected, x, a, mu, label, bce_per_brush, optimizer, n_iter, scheduler=None):
        for _ in range(n_iter):
            loss_per_brush = [
                bce_per_brush[t](self.predict(x, a[t], mu[t]), label[t])
                for t, _ in enumerate(selected)
            ]
            smoothness_loss = self._calculate_smoothness_loss(a, mu, len(selected))
            total_loss = sum(loss_per_brush) + smoothness_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

    def _calculate_smoothness_loss(self, a, mu, n_selected):
        if self.smoothness_scale == 0:
            return 0
        if n_selected == 2:
            return self.smoothness_scale * (
                _SMOOTHNESS_A_PAIR * (a[1:] - a[:-1]).pow(2).mean()
                + _SMOOTHNESS_MU_PAIR * (mu[1:] - mu[:-1]).pow(2).mean()
            )
        if n_selected > 2:
            return self.smoothness_scale * (
                _SMOOTHNESS_A_MULTI * (a[1:] - a[:-1]).pow(2).mean()
                + _SMOOTHNESS_MU_MULTI * (mu[1:] - mu[:-1]).pow(2).mean()
            )
        return 0

    def _generate_predicates(self, selected, x0, a, mu, mean, scale, vmin, vmax, attribute_names):
        predicates = []
        for t, st in enumerate(selected):
            r = 1 / a[t].abs()
            predicate_clauses = []
            for k in range(x0.shape[1]):
                r_k = (r[k] * scale[k]).item()
                mu_k = (mu[t, k] * scale[k] + mean[k]).item()
                ci = [mu_k - r_k, mu_k + r_k]
                if ci[0] >= ci[1]:
                    continue
                ci[0] = max(ci[0], vmin[k])
                ci[1] = min(ci[1], vmax[k])
                if not (ci[0] <= vmin[k] and ci[1] >= vmax[k]):
                    predicate_clauses.append(dict(dim=k, interval=ci, attribute=attribute_names[k]))
            predicates.append(predicate_clauses)
        return predicates
