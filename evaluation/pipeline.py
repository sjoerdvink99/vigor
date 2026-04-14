import os
import random
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.feature_selection import f_classif
from sklearn.metrics import f1_score as _f1_score
from src import Graph, Predicate, PredicateInduction


_ASYMMETRIC_CONFUSION = {
    "NODELINK": "NODELINK_POSITIONING",
    "NODELINK_POSITIONING": "NODELINK_FACETING",
    "NODELINK_FACETING": "NODELINK",
    "MATRIX": "BIOFABRIC",
    "BIOFABRIC": "MATRIX",
    "QUILTS": "NODELINK_POSITIONING",
    "TREEMAP": "SUNBURST",
    "SUNBURST": "TREEMAP",
}


def _simulate_attributes(H):
    n_node_types = random.randint(0, 5)
    n_node_attrs = random.randint(0, 15)
    n_edge_types = random.randint(0, 4)
    n_edge_attrs = random.randint(0, 10)
    spatial = random.random() < 0.2
    for node in H.nodes():
        if n_node_types > 0:
            H.nodes[node]["label"] = str(random.randint(0, n_node_types - 1))
        for i in range(n_node_attrs):
            H.nodes[node][f"a{i}"] = random.random()
        if spatial:
            H.nodes[node]["lat"] = random.uniform(-90, 90)
            H.nodes[node]["lon"] = random.uniform(-180, 180)
    for u, v in H.edges():
        if n_edge_types > 0:
            H[u][v]["type"] = str(random.randint(0, n_edge_types - 1))
        for i in range(n_edge_attrs):
            H[u][v][f"e{i}"] = random.random()


def _make_ba(nodes_min, nodes_max):
    n = np.random.randint(max(nodes_min, 4), nodes_max + 1)
    m = np.random.randint(1, max(2, n // 4))
    return nx.barabasi_albert_graph(n, m)


def _make_ws(nodes_min, nodes_max):
    n = np.random.randint(max(nodes_min, 6), nodes_max + 1)
    k = np.random.randint(2, max(3, n // 3)) * 2
    p = np.random.uniform(0.05, 0.4)
    return nx.watts_strogatz_graph(n, k, p)


def _make_sbm(nodes_min, nodes_max):
    n_communities = np.random.randint(2, 6)
    total = np.random.randint(max(nodes_min, n_communities * 3), nodes_max + 1)
    sizes = np.diff(np.sort(np.random.choice(total - 1, n_communities - 1, replace=False)))
    sizes = np.concatenate([[sizes[0]], sizes[1:], [total - sizes.sum()]])
    sizes = np.maximum(sizes, 1).tolist()
    p_in = np.random.uniform(0.2, 0.7)
    p_out = np.random.uniform(0.01, 0.1)
    probs = [[p_in if i == j else p_out for j in range(len(sizes))] for i in range(len(sizes))]
    return nx.stochastic_block_model(sizes, probs)


def generate_graphs(n_graphs, nodes_min=2, nodes_max=100, file_path=None, families=None):
    file_exists = file_path is not None and os.path.exists(file_path)
    graphs = []
    while len(graphs) < n_graphs:
        rand_val = random.random()
        family = None
        try:
            if rand_val < 0.08:
                family = 'tree'
                H = nx.balanced_tree(np.random.randint(2, 4), np.random.randint(2, 5))
                if H.number_of_nodes() > nodes_max:
                    continue
            elif rand_val < 0.13:
                family = 'tree'
                n = np.random.randint(max(nodes_min, 10), min(80, nodes_max) + 1)
                H = nx.generators.trees.random_labeled_tree(n)
            elif rand_val < 0.18:
                family = 'cycle'
                H = nx.cycle_graph(np.random.randint(nodes_min, min(30, nodes_max)))
            elif rand_val < 0.28:
                family = 'ba'
                H = _make_ba(nodes_min, nodes_max)
            elif rand_val < 0.36:
                family = 'ws'
                H = _make_ws(nodes_min, nodes_max)
            elif rand_val < 0.44:
                family = 'sbm'
                H = _make_sbm(nodes_min, nodes_max)
            else:
                family = 'gnp'
                directed = random.random() < 0.3
                H = nx.fast_gnp_random_graph(
                    np.random.randint(nodes_min, nodes_max),
                    np.random.uniform(0, 0.5),
                    directed=directed,
                )
        except Exception:
            continue

        if families is not None and family not in families:
            continue

        _simulate_attributes(H)
        G = Graph()
        G.from_existing_graph(H)
        try:
            statistics = G.get_statistics()
            graphs.append(statistics)
            if file_path is not None:
                df = pd.DataFrame([statistics])
                df.to_csv(file_path, mode="a", header=not file_exists, index=False)
                file_exists = True
        except Exception:
            pass

    return pd.DataFrame(graphs)


def label_graphs(df, predicates, conformance=1.0, mode="probability", asymmetric_noise=False):
    if not predicates:
        return df

    if mode == "probability":
        vistype_clauses = {}
        for vistype, attr, minval, maxval in predicates:
            if attr in df.columns:
                if vistype.name not in vistype_clauses:
                    vistype_clauses[vistype.name] = {}
                vistype_clauses[vistype.name][attr] = (minval, maxval)
        score_dict = {}
        for vis, clauses in vistype_clauses.items():
            total = pd.Series(0.0, index=df.index)
            for attr, (lo, hi) in clauses.items():
                mu = (lo + hi) / 2.0
                a = 2.0 / max(hi - lo, 0.01)
                total += (a * (df[attr] - mu)).abs() ** 3
            score_dict[vis] = 1.0 / (1.0 + total / max(len(clauses), 1))
        scores = pd.DataFrame(score_dict)
        predicted_labels = scores.idxmax(axis=1)
    elif mode == "conjunction":
        vistype_clauses = {}
        for vistype, attr, minval, maxval in predicates:
            if attr in df.columns:
                if vistype.name not in vistype_clauses:
                    vistype_clauses[vistype.name] = {}
                vistype_clauses[vistype.name][attr] = [minval, maxval]
        vistype_predicates = {k: [Predicate(clauses=v)] for k, v in vistype_clauses.items()}
        for v in vistype_predicates.values():
            v[0].fit(df)
        vistype_labels = {k: pd.DataFrame({p.attrs[0]: p.mask for p in v}) for k, v in vistype_predicates.items()}
        scores = pd.DataFrame({k: v.sum(axis=1) for k, v in vistype_labels.items()})
        predicted_labels = scores.idxmax(axis=1)
    else:
        vistype_predicates = {}
        for vistype, attr, minval, maxval in predicates:
            if attr in df.columns:
                predicate = Predicate(clauses={attr: [minval, maxval]})
                predicate.fit(df)
                if vistype.name not in vistype_predicates:
                    vistype_predicates[vistype.name] = []
                vistype_predicates[vistype.name].append(predicate)
        vistype_labels = {k: pd.DataFrame({p.attrs[0]: p.mask for p in v}) for k, v in vistype_predicates.items()}
        scores = pd.DataFrame({k: v.sum(axis=1) for k, v in vistype_labels.items()})
        predicted_labels = scores.idxmax(axis=1)

    unique_labels = list(scores.columns)

    def _apply_noise(pred):
        if np.random.random() <= conformance:
            return pred
        if asymmetric_noise and pred in _ASYMMETRIC_CONFUSION:
            target = _ASYMMETRIC_CONFUSION[pred]
            return target if target in unique_labels else np.random.choice(unique_labels)
        return np.random.choice(unique_labels)

    return predicted_labels.apply(_apply_noise)


def _get_predicate(vigor, X, y, n_iter, eps, balanced, prior=None, n_restarts=1,
                   random_init=False):
    best_pred = None
    best_f1 = -1.0
    for i in range(n_restarts):
        result = vigor.compute_predicate_sequence(
            X.values,
            y[None],
            attribute_names=list(X.columns),
            n_iter=n_iter,
            eps=eps,
            balanced=balanced,
            priors=[prior],
            init_noise=0.0 if i == 0 else 0.3,
            random_init=random_init and i > 0,
        )
        p = Predicate(result[0])
        p.fit(X)
        score = _f1_score(y.astype(int), p.mask.astype(int), zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_pred = p
    return best_pred


def _denormalize(pred, min_vals, max_vals):
    for attr, (lo, hi) in pred.clauses.items():
        orig_min = min_vals[attr]
        orig_max = max_vals[attr]
        pred.clauses[attr] = [
            lo * (orig_max - orig_min) + orig_min,
            hi * (orig_max - orig_min) + orig_min,
        ]
    return pred


def _normalize_prior(prior, min_vals, max_vals):
    if prior is None:
        return None
    normalized = {}
    for attr, (lo, hi) in prior.clauses.items():
        if attr in min_vals.index:
            span = max_vals[attr] - min_vals[attr] + 1e-8
            normalized[attr] = [
                (lo - min_vals[attr]) / span,
                (hi - min_vals[attr]) / span,
            ]
    return normalized if normalized else None


def learn_predicates(df, labels, label_names=None, n_iter=2000, eps=1e-4, balanced=True,
                     initial_predicates=None, n_restarts=2, top_k=12, model_kwargs=None):
    df = df.loc[:, df.nunique() > 1]
    min_vals = df.min()
    max_vals = df.max()
    normalized = (df - min_vals) / (max_vals - min_vals + 1e-8)

    _mk = {k: v for k, v in (model_kwargs or {}).items() if k != "random_init"}
    vigor = PredicateInduction(**_mk)
    label_names = labels.unique() if label_names is None else label_names
    learned = {}
    for vis in label_names:
        y = (labels == vis).values
        if y.sum() < 2:
            continue

        if top_k is not None and top_k < normalized.shape[1]:
            f_stats, _ = f_classif(normalized, y)
            f_stats = np.nan_to_num(f_stats, nan=0.0)
            top_cols = normalized.columns[np.argsort(f_stats)[::-1][:top_k]]
            X_vis = normalized[top_cols]
        else:
            X_vis = normalized

        prior = _normalize_prior(
            initial_predicates.get(vis) if initial_predicates else None,
            min_vals, max_vals
        )
        random_init = (model_kwargs or {}).get("random_init", True)
        pred = _get_predicate(vigor, X_vis, y, n_iter, eps, balanced, prior=prior,
                              n_restarts=n_restarts, random_init=random_init)
        pred = _denormalize(pred, min_vals, max_vals)
        learned[vis] = pred

    return learned
