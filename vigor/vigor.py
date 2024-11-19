import numpy as np
from typing import Dict, List, Tuple, Union
from vigor.visualization import Visualization
from vigor.visualization_types import VisualizationType
import torch
from torch import nn
from torch import optim

device = "cuda" if torch.cuda.is_available() else "cpu"

class VIGOR:
    def __init__(self) -> None:
        self.visualizations = []

    def add_visualization(self, predicate_set: Visualization) -> None:
        self.visualizations.append(predicate_set)

    def recommend(self, stats: Dict[str, float], n: int = 1) -> Union[Tuple[VisualizationType, float], List[Tuple[VisualizationType, float]]]:
        scores = [vs.compute_relevance(stats) for vs in self.visualizations]
        scores = np.array(scores)

        if n == 1:
            best_index = np.argmax(scores)
            return self.visualizations[best_index].visualization_type, scores[best_index]
        else:
            best_indices = np.argsort(scores)[-n:][::-1]
            return [(self.visualizations[i].visualization_type, scores[i]) for i in best_indices]
    
    def predict(self, x, a, mu):
        b = 4
        pred = 1 / (1 + ((a.abs() * (x - mu).abs()).pow(b)).sum(1))
        return pred

    def compute_predicate_sequence(
            self,
            x0,
            selected,
            attribute_names=[],
            n_iter=1000,
        ):
        """
        x0 - numpy array, shape=[n_points, n_feature]. Data points
        selected - boolean array. shape=[brush_index, n_points] of selection
        """

        n_points, n_features = x0.shape

        # prepare training data
        # orginal data extent
        vmin = x0.min(0)
        vmax = x0.max(0)
        x = torch.from_numpy(x0.astype(np.float32)).to(device)
        label = torch.from_numpy(selected).float().to(device)
        # normalize
        mean = x.mean(0)
        scale = x.std(0) + 1e-2
        x = (x - mean) / scale

        # Trainable parameters
        # since data is normalized,
        # mu can initialized around mean_pos examples
        # a can initialized around a constant across all axes
        selection_centroids = torch.stack([x[sel_t].mean(0) for sel_t in selected], 0)
        selection_std = torch.stack([x[sel_t].std(0) for sel_t in selected], 0)

        # initialize the bounding box center (mu) at the data centroid, +-0.1 at random
        mu_init = selection_centroids
        a_init = 1 / selection_std
        a = a_init.to(device)
        mu = mu_init.to(device)
        a.requires_grad_(True)
        mu.requires_grad_(True)

        # For each brush,
        # weight-balance selected vs. unselected based on their size
        # and create a weighted BCE loss function (for each brush)
        bce_per_brush = []
        for st in selected:  # for each brush, define their class-balanced loss function
            n_selected = st.sum()  # st is numpy array
            n_unselected = n_points - n_selected
            instance_weight = torch.ones(x.shape[0]).to(device)
            instance_weight[st] = n_points / n_selected
            instance_weight[~st] = n_points / n_unselected
            bce = nn.BCELoss(weight=instance_weight)
            bce_per_brush.append(bce)

        optimizer = optim.SGD(
            [
                {"params": mu, "weight_decay": 0},
                # smaller a encourages larger range of the bounding box
                {"params": a, "weight_decay": 0.25},
            ],
            lr=1e-2,
            momentum=0.4,
            nesterov=True,
        )

        # training loop
        for e in range(n_iter):
            loss_per_brush = []
            for t, st in enumerate(selected):  # for each brush, compute loss
                # use all selected data
                # randomly sample unselected data with similar size
                pred = self.predict(x, a[t], mu[t])
    #             print(pred)
                loss = bce(pred, label[t])
                # loss += (mu[t] - selection_centroids[t]).pow(2).mean() * 20
                loss_per_brush.append(loss)
                smoothness_loss = 0
                if len(selected) == 2:
                    smoothness_loss += 5 * (a[1:] - a[:-1]).pow(2).mean()
                    smoothness_loss += 1 * (mu[1:] - mu[:-1]).pow(2).mean()
                elif len(selected) > 2:
                    smoothness_loss += 500 * (a[1:] - a[:-1]).pow(2).mean()
                    smoothness_loss += 10 * (mu[1:] - mu[:-1]).pow(2).mean()

            total_loss = sum(loss_per_brush) + smoothness_loss  # + sparsity_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            if e % max(1, (n_iter // 10)) == 0:
                print(f"[{e:>4}] loss {loss.item()}")
        a.detach_()
        mu.detach_()
            

        # predicate clause selection
        # r is the range of the bounding box on each dimension
        # bounding box is defined by the level set of prediction=0.5
        predicates = []
        # for each brush, generate a predicate from a[t] and mu[t]
        for t, st in enumerate(selected):
            r = 1 / a[t].abs()
            predicate_clauses = []
            for k in range(n_features):  # for each attribute
                vmin_selected = x0[st, k].min()
                vmax_selected = x0[st, k].max()
                # denormalize
                r_k = (r[k] * scale[k]).item()
                mu_k = (mu[t, k] * scale[k] + mean[k]).item()
                ci = [mu_k - r_k, mu_k + r_k]
                assert ci[0] < ci[1], "ci[0] is not less than ci[1]"
                if ci[0] < vmin[k]:
                    ci[0] = vmin[k]
                if ci[1] > vmax[k]:
                    ci[1] = vmax[k]

                # feature selection based on extent range
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