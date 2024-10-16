import numpy as np
from typing import Dict, List
from vigor.model.visualization import Visualization
from vigor.visualization_types import VisualizationType

class VIGOR:
  def __init__(self) -> None:
    self.visualizations = []

  def add_visualization(self, predicate_set: Visualization) -> None:
    self.visualizations.append(predicate_set)

  def recommend(self, stats: Dict[str, float]) -> VisualizationType:
    scores = []

    for vs in self.visualizations:
        score = vs.compute_score(stats)
        scores.append(score)

    scores = np.array(scores)
    best_index = np.argmax(scores)
    return self.visualizations[best_index].visualization_type
  
  def recommend_n(self, stats: Dict[str, float], n: int) -> List[VisualizationType]:
    scores = []

    for vs in self.visualizations:
        score = vs.compute_score(stats)
        scores.append(score)

    scores = np.array(scores)
    best_indices = np.argsort(scores)[-n:][::-1]
    return [self.visualizations[i].visualization_type for i in best_indices]