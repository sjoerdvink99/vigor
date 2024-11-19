import numpy as np
import torch
from torch import nn, optim

device = "cuda" if torch.cuda.is_available() else "cpu"

class VIGOR:
    def __init__(self) -> None:
        self.visualizations = []

    def predict(self, x, a, mu):
        b = 4
        pred = 1 / (1 + ((a.abs() * (x - mu).abs()).pow(b)).sum(1))
        return pred

    def compute_predicate_sequence(self, x0, selected, attribute_names=[], n_iter=1000):
        n_points, n_features = x0.shape

        vmin = x0.min(0)
        vmax = x0.max(0)
        x = torch.from_numpy(x0.astype(np.float32)).to(device)
        label = torch.from_numpy(selected).float().to(device)

        mean = x.mean(0)
        scale = x.std(0) + 1e-2
        x = (x - mean) / scale

        selection_centroids = torch.stack([x[sel_t].mean(0) for sel_t in selected], 0)
        selection_std = torch.stack([x[sel_t].std(0) for sel_t in selected], 0)

        mu_init = selection_centroids
        a_init = 1 / selection_std
        a = a_init.to(device)
        mu = mu_init.to(device)
        a.requires_grad_(True)
        mu.requires_grad_(True)

        bce_per_brush = []
        for st in selected:
            n_selected = st.sum()
            n_unselected = n_points - n_selected
            instance_weight = torch.ones(x.shape[0]).to(device)
            instance_weight[st] = n_points / n_selected
            instance_weight[~st] = n_points / n_unselected
            bce = nn.BCELoss(weight=instance_weight)
            bce_per_brush.append(bce)

        optimizer = optim.SGD(
            [
                {"params": mu, "weight_decay": 0},
                {"params": a, "weight_decay": 0.25},
            ],
            lr=1e-2,
            momentum=0.4,
            nesterov=True,
        )

        for e in range(n_iter):
            loss_per_brush = []
            for t, st in enumerate(selected):
                pred = self.predict(x, a[t], mu[t])
                loss = bce_per_brush[t](pred, label[t])
                loss_per_brush.append(loss)

                smoothness_loss = 0
                if len(selected) == 2:
                    smoothness_loss += 5 * (a[1:] - a[:-1]).pow(2).mean()
                    smoothness_loss += 1 * (mu[1:] - mu[:-1]).pow(2).mean()
                elif len(selected) > 2:
                    smoothness_loss += 500 * (a[1:] - a[:-1]).pow(2).mean()
                    smoothness_loss += 10 * (mu[1:] - mu[:-1]).pow(2).mean()

            total_loss = sum(loss_per_brush) + smoothness_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            if e % max(1, (n_iter // 10)) == 0:
                print(f"[{e:>4}] loss {total_loss.item()}")

        a.detach_()
        mu.detach_()

        predicates = []
        for t, st in enumerate(selected):
            r = 1 / a[t].abs()
            predicate_clauses = []
            for k in range(n_features):
                vmin_selected = x0[st, k].min()
                vmax_selected = x0[st, k].max()
                r_k = (r[k] * scale[k]).item()
                mu_k = (mu[t, k] * scale[k] + mean[k]).item()
                ci = [mu_k - r_k, mu_k + r_k]
                assert ci[0] < ci[1], "ci[0] is not less than ci[1]"
                if ci[0] < vmin[k]:
                    ci[0] = vmin[k]
                if ci[1] > vmax[k]:
                    ci[1] = vmax[k]

                should_include = not (ci[0] <= vmin[k] and ci[1] >= vmax[k])

                if should_include:
                    if ci[0] < vmin_selected:
                        ci[0] = vmin_selected
                    if ci[1] > vmax_selected:
                        ci[1] = vmax_selected
                    predicate_clauses.append(
                        dict(
                            dim=k,
                            interval=ci,
                            attribute=attribute_names[k],
                        )
                    )
            predicates.append(predicate_clauses)
        return predicates