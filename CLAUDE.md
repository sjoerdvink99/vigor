# CLAUDE.md

## Project

Research codebase for learning graph visualization design guidelines as predicates. Graphs are embedded as feature vectors of statistical properties. Predicates define axis-aligned regions in this feature space, expressing conditions under which a visualization type is appropriate. The induction algorithm optimizes predicate boundaries from labeled graph-visualization pairs using gradient-based optimization with PyTorch.

Paper: *Graph Visualization Design Guidelines as Learnable Predicates* (Best Paper, GRIVAPP 2025).

## Structure

```
vigor/
├── src/                       # Core library
│   ├── __init__.py            # Public API: Graph, PredicateInduction, Predicate, predicates, nobre_predicates
│   ├── model.py               # Predicate induction algorithm (PyTorch SGD)
│   ├── graph.py               # Graph dataclass with 30+ statistics via NetworkX
│   ├── predicate.py           # Predicate class (interval constraints as conjunctions)
│   ├── guidelines.py          # Literature-derived predicate sets
│   ├── types.py               # VisualizationType and NobreVisualizations enums
│   └── utils.py               # Spatial/temporal attribute keyword detection
├── evaluation/
│   ├── pipeline.py            # generate_graphs, label_graphs, learn_predicates
│   ├── metrics.py             # compute_metrics, accuracy_metrics
│   └── run.py                 # CLI entry point
├── tests/
│   ├── test_graph.py
│   ├── test_predicate.py
│   └── test_model.py
├── data/
│   └── generated_graphs_example.csv
└── docs/
    └── GRIVAPP_*.pdf
```

## Key Concepts

**Feature space**: Each graph is a point in R^M where each dimension is a graph statistic. Extracted by `Graph.get_statistics()`, covering general properties, connectivity, cohesion, and element counts.

**Predicate**: A conjunction of interval constraints over graph statistics. Example: `Predicate(clauses={'density': [0, 0.1], 'n_nodes': [0, 100]})` evaluates to true only when all clauses are satisfied.

**Probability function** (Equation 1 in paper): `Pr(Φ, s) = 1 / (1 + sum_j |a_j * (s_j - mu_j)|^3)` where `mu` is the predicate center and `a` controls per-dimension sensitivity (inverse radius). Exponent b=3 per paper.

**Induction algorithm** (`model.py`): For each visualization type, optimizes `(a, mu)` via SGD with Nesterov momentum and weight decay on `a`. A smoothness regularizer penalizes abrupt changes across brushes. Learned parameters are converted to intervals as `mu ± 1/|a|` in original data scale.

**Predicate sets** (`src/guidelines.py`): Two sets. `predicates` covers 7 types (NODELINK, MATRIX, NODETRIX, NODELINK_MAP, PAOHVIS, CHORD_DIAGRAM, TREEMAP). `nobre_predicates` covers 8 types from Nobre et al. (2019).

## Evaluation Pipeline

Three simulated user profiles:

| Profile | Conformance |
|---|---|
| Informed | 1.00 |
| Semi-informed | 0.75 |
| Uninformed | 0.50 |

`label_graphs` assigns each graph a visualization type by scoring how many predicate clauses it satisfies per type (additive). Conformance controls how often the score-maximizing label is assigned vs a random one.

`compute_metrics` returns three metric categories per visualization type:
- `exact`: IoU, deviation, inclusion between learned and ground-truth intervals per statistic
- `describe`: precision, recall, F1 on training data
- `generalize`: precision, recall, F1 on held-out test data

## Stack

- Python 3.12, uv for dependency management
- PyTorch (SGD optimization)
- NetworkX (graph statistics)
- pandas, numpy, scikit-learn
- pytest for tests

## Commands

```bash
uv sync                                              # install dependencies
uv run pytest tests/                                 # run tests
uv run python -m evaluation.run --experiment recovery
uv run python -m evaluation.run --experiment nobre --output results.json
```
