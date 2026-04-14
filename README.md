# Graph Visualization Design Guidelines as Learnable Predicates

This codebase represents graph visualization design guidelines as predicates over graph statistics and learns them from labeled data. Each graph is embedded as a feature vector in a statistical feature space. A predicate defines a bounded, axis-aligned region in this space, capturing the conditions on graph properties (density, node count, clustering coefficient, etc.) under which a specific visualization technique is most appropriate. The induction algorithm learns these regions from labeled graph-visualization pairs using gradient-based optimization.

This code accompanies the paper *Graph Visualization Design Guidelines as Learnable Predicates*, which received the Best Paper Award at GRIVAPP 2025.

## Install

Requires Python 3.12+.

```bash
uv sync
```

## Usage

Extract statistics from a graph:

```python
import networkx as nx
from src import Graph

H = nx.fast_gnp_random_graph(50, 0.1)
G = Graph()
G.from_existing_graph(H)
stats = G.get_statistics()
```

Learn predicates from labeled graphs:

```python
from src.guidelines import predicates
from evaluation.pipeline import generate_graphs, label_graphs, learn_predicates

df = generate_graphs(1000)
labels = label_graphs(df, predicates, conformance=1.0, mode="probability")
learned = learn_predicates(df, labels)
```

Evaluate learned predicates against ground-truth intervals:

```python
from evaluation.metrics import compute_metrics

metrics = compute_metrics(predicates, learned, train_df, test_df, train_labels, test_labels)
```

## Evaluation

Run the full evaluation from the command line:

```bash
uv run python -m evaluation.run recovery   # Rule recovery — IoU between initial and learned predicates (§7.1)
uv run python -m evaluation.run e1         # Personalization — noise robustness and sample efficiency (§7.2)
uv run python -m evaluation.run e2         # Real-world domain classification on TU datasets
uv run python -m evaluation.run all        # Run all three experiments
```

The `e1` experiment simulates three user profiles with different adherence to expert guidelines:

| Profile | Conformance |
|---|---|
| Informed | 1.00 |
| Semi-informed | 0.75 |
| Uninformed | 0.50 |

Results and figures are saved to `results/`.

## Tests

```bash
uv run pytest tests/
```

## Citation

```bibtex
@inproceedings{vink2025predicates,
  title     = {Graph Visualization Design Guidelines as Learnable Predicates},
  author    = {Vink, Sjoerd and Montambault, Brian and Li, Mingwei and Chang, Remco and Behrisch, Michael},
  booktitle = {GRIVAPP},
  year      = {2025}
}
```
