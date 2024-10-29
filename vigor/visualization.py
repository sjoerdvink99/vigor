import numpy as np
from typing import Dict
import torch
import torch.nn as nn
import torch.optim as optim
from vigor.visualization_types import VisualizationType
from vigor.predicate import Predicate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Visualization:
    def __init__(self, visualization_type: VisualizationType) -> None:
        self.visualization_type = visualization_type
        self.predicates = []

    def add_predicate(self, predicate: Predicate) -> None:
        self.predicates.append(predicate)

    def compute_relevance(self, stats: Dict[str, float]) -> float:
        """Compute total score based on the evaluation of predicates."""
        scores = np.array([predicate.relevance(stats.get(predicate.statistic, 0)) for predicate in self.predicates])
        return np.sum(scores) / len(self.predicates)

    def update(self, feedback: int, stats: Dict[str, float], n_iter: int = 1000):
        """Update predicate scores using feedback as reward"""
        x = torch.from_numpy(np.array([[stats[p.statistic] for p in self.predicates]])).float().to(device)
        x = torch.cat([x, torch.rand(x.shape[0], 1)], dim=1)
        feedback_tensor = torch.tensor([feedback], dtype=torch.float32, device=device).view(1, 1)

        a = torch.tensor([p.a for p in self.predicates], requires_grad=True, device=device)
        mu = torch.tensor([p.mu for p in self.predicates], requires_grad=True, device=device)

        optimizer = optim.SGD([a, mu], lr=5e-3, momentum=0.3, nesterov=True)
        bce_loss = nn.BCELoss()

        for _ in range(n_iter):
            pred = self._predict(x, a, mu)
            loss = bce_loss(pred, feedback_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self._apply_learned_thresholds(a, mu)

    def _apply_learned_thresholds(self, a: torch.Tensor, mu: torch.Tensor) -> None:
        """Adjust predicates based on learned a and mu."""
        for i, predicate in enumerate(self.predicates):
            predicate.mu = mu[i].item()
            predicate.a = a[i].item()

    def _predict(self, x: torch.Tensor, a: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """UMAP-inspired prediction function based on `a` and `mu`."""
        b = 5
        pred = 1 / (1 + ((a.abs() * (x - mu).abs()).pow(b)).sum(1, keepdim=True))
        return pred