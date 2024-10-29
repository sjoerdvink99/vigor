import numpy as np
from typing import Dict, List, Tuple, Union
from vigor.visualization import Visualization
from vigor.visualization_types import VisualizationType

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