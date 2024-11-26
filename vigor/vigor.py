import numpy as np
import torch
from torch import nn, optim
from textwrap import dedent

device = "cuda" if torch.cuda.is_available() else "cpu"

class VIGOR:
    def __init__(self) -> None:
        self.visualizations = []

    def predict(self, x, a, mu):
        b = 4
        pred = 1 / (1 + ((a.abs() * (x - mu).abs()).pow(b)).sum(1))
        return pred

    def compute_predicate_sequence(self, x0, selected, attribute_names=[], n_iter=1000, eps=1e-2, balanced=True):
        n_points, n_features = x0.shape

        vmin, vmax = x0.min(0), x0.max(0)
        x = torch.from_numpy(x0.astype(np.float32)).to(device)
        label = torch.from_numpy(selected).float().to(device)

        mean, scale = x.mean(0), x.std(0) + eps
        x = (x - mean) / scale

        selection_centroids = torch.stack([x[sel_t].mean(0) for sel_t in selected], 0)
        selection_std = torch.stack([x[sel_t].std(0) for sel_t in selected], 0) + eps

        a, mu = (1 / selection_std).to(device), selection_centroids.to(device)
        
        a.requires_grad_(True)
        mu.requires_grad_(True)

        bce_per_brush = self._create_bce_per_brush(selected, n_points, x, balanced=balanced)

        optimizer = optim.SGD(
            [{"params": mu, "weight_decay": 0}, {"params": a, "weight_decay": 0.25}],
            lr=1e-2,
            momentum=0.4,
            nesterov=True,
        )
        
        try:
            self._optimize(selected, x, a, mu, label, bce_per_brush, optimizer, n_iter)
            failed = False
        except:
            failed = True
            
        a.detach_()
        mu.detach_()
        if failed:
            return failed, (mu, a)
        else:
            return failed, self._generate_predicates(selected, x0, a, mu, mean, scale, vmin, vmax, attribute_names, n_points, x, label)

    def _create_bce_per_brush(self, selected, n_points, x, balanced=True):
        bce_per_brush = []
        for st in selected:
            n_selected = st.sum()
            n_unselected = n_points - n_selected
            instance_weight = torch.ones(x.shape[0]).to(device)
            if balanced:
                instance_weight *= (selected[0]*(n_unselected/n_points) + (1-selected[0])*(n_selected/n_points))
            bce_per_brush.append(nn.BCELoss(weight=instance_weight))
        return bce_per_brush

    def _optimize(self, selected, x, a, mu, label, bce_per_brush, optimizer, n_iter):
        for e in range(n_iter):
            loss_per_brush = []
            for t, st in enumerate(selected):
                pred = self.predict(x, a[t], mu[t])
                loss = bce_per_brush[t](pred, label[t])
                loss_per_brush.append(loss)

                smoothness_loss = self._calculate_smoothness_loss(a, mu, len(selected))

            total_loss = sum(loss_per_brush) + smoothness_loss
            optimizer.zero_grad()
            total_loss.backward()
            
            optimizer.step()
            if e % max(1, (n_iter // 10)) == 0:
                print(f"[{e:>4}] loss {total_loss.item()}")

    def _calculate_smoothness_loss(self, a, mu, n_selected):
        smoothness_loss = 0
        if n_selected == 2:
            smoothness_loss += 5 * (a[1:] - a[:-1]).pow(2).mean()
            smoothness_loss += 1 * (mu[1:] - mu[:-1]).pow(2).mean()
        elif n_selected > 2:
            smoothness_loss += 500 * (a[1:] - a[:-1]).pow(2).mean()
            smoothness_loss += 10 * (mu[1:] - mu[:-1]).pow(2).mean()
        return smoothness_loss

    def _generate_predicates(self, selected, x0, a, mu, mean, scale, vmin, vmax, attribute_names, n_points, x, label):
        qualities = []
        for t, st in enumerate(selected):  # for each brush, compute quality
            pred = self.predict(x, a[t], mu[t])
            pred = (pred > 0.5).float()
            correct = (pred == label[t]).float().sum().item()
            total = n_points
            accuracy = correct / total
            # 1 meaning points are selected
            tp = ((pred == 1).float() * (label == 1).float()).sum().item()
            fp = ((pred == 1).float() * (label == 0).float()).sum().item()
            fn = ((pred == 0).float() * (label == 1).float()).sum().item()
            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            f1 = 2 / (1 / precision + 1 / recall) if precision > 0 and recall > 0 else 0
            print(
                dedent(
                    f"""
                brush = {t}
                accuracy = {accuracy}
                precision = {precision}
                recall = {recall}
                f1 = {f1}
            """
                )
            )
            qualities.append(
                dict(brush=t, accuracy=accuracy, precision=precision, recall=recall, f1=f1)
            )

        predicates = []
        for t, st in enumerate(selected):
            r = 1 / a[t].abs()
            predicate_clauses = []
            for k in range(x0.shape[1]):
                vmin_selected, vmax_selected = x0[st, k].min(), x0[st, k].max()
                r_k = (r[k] * scale[k]).item()
                mu_k = (mu[t, k] * scale[k] + mean[k]).item()
                ci = [mu_k - r_k, mu_k + r_k]
                assert ci[0] < ci[1], "ci[0] is not less than ci[1]"
                ci[0] = max(ci[0], vmin[k])
                ci[1] = min(ci[1], vmax[k])

                if not (ci[0] <= vmin[k] and ci[1] >= vmax[k]):
                    ci[0] = max(ci[0], vmin_selected)
                    ci[1] = min(ci[1], vmax_selected)
                    predicate_clauses.append(dict(dim=k, interval=ci, attribute=attribute_names[k]))
            predicates.append(predicate_clauses)
        return predicates