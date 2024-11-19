import random
import numpy as np
import pandas as pd
import networkx as nx
from vigor import Graph, Predicate, VIGOR

def generate_graphs(n_graphs, nodes_min=2, nodes_max=200):
    """
    Function to generate a list of graphs with random number of nodes and edges
    """
    graphs = []
    for i in range(n_graphs):
        n = np.random.randint(nodes_min, nodes_max)
        p = np.random.uniform(0, 0.5)
        G = Graph()
        G.from_existing_graph(nx.fast_gnp_random_graph(n, p))

        try:
            statistics = G.get_statistics(testing=True)
            print("Generated statistics for graph", i, statistics)
            graphs.append(statistics)
        except:
            print(f"Graph {i} failed to extract statistics")

    df = pd.DataFrame(graphs)
    return df
    
def label_graphs(df, predicates, conformance=1):
    """
    Function to label graphs based on predicates
    """
    if not predicates:
        print("No predicates provided")
        return df
    
    vistype_predicates = {}
    for vistype, attr, minval, maxval in predicates:
        if attr in df.columns:
            predicate = Predicate(clauses={attr: [minval, maxval]})
            predicate.fit(df)
            if vistype.name in vistype_predicates:
                vistype_predicates[vistype.name].append(predicate)
            else:
                vistype_predicates[vistype.name] = [predicate]
    
    vistype_labels = {k: pd.DataFrame({p.attrs[0]: p.mask for p in v}) for k, v in vistype_predicates.items()}
    scores = pd.DataFrame({k: v.sum(axis=1) for k,v in vistype_labels.items()})
    predicted_labels = scores.idxmax(axis=1)

    unique_labels = list(scores.columns)
    final_labels = predicted_labels.apply(
        lambda pred: pred if random.random() <= conformance else random.choice(unique_labels)
    )
    
    return final_labels

def get_predicates(vigor, X, y, n_iter=1000):
    print("get_predicates", X.values, y[None], X.columns)
    predicates = vigor.compute_predicate_sequence(
        X.values,
        y[None],
        attribute_names=X.columns,
        n_iter=n_iter,
    )
    
    p = Predicate(predicates[0])
    p.fit(X)
    return p

def learn_predicates(df, labels, n_iter=1000):
    """
    Function to learn predicates from the data
    """
    df = df.loc[:, df.nunique() > 1]
    epsilon = 1e-8  # Small value to avoid division by zero
    graphs_normalized = (df - df.min()) / (df.max() - df.min() + epsilon)

    vigor = VIGOR()
    pred_list = {}

    for visualization in labels.unique():
        ypos = (labels == visualization).values
        yneg = ~ypos
        print(f"Learning predicates for {visualization}")
        pred_pos = get_predicates(vigor, graphs_normalized, ypos, n_iter=n_iter)
        pred_neg = get_predicates(vigor, graphs_normalized, yneg, n_iter=n_iter)

        pred_list[visualization] = pred_pos, pred_neg

    return pred_list

def compute_metrics(initial, learned):
    """
    Compute Intersection-over-Union (IoU), Deviation, and Inclusion metrics for evaluation
    Intersection-over-Union: Measures the overlap between the initial and learned ranges
    Deviation: Absolute deviation between the initial and learned predicate ranges
    Inclusion: Checks if the learned range fully falls within the initial range or vice versa
    """
    metrics = []
    for init, learn in zip(initial, learned):
        min_init, max_init = init['min'], init['max']
        min_learn, max_learn = learn['min'], learn['max']

        intersection = max(0, min(max_init, max_learn) - max(min_init, min_learn))
        union = max(max_init, max_learn) - min(min_init, min_learn)
        iou = intersection / union if union > 0 else 0
        deviation = (abs(min_init - min_learn) + abs(max_init - max_learn)) / 2

        inclusion = 1 if (
            (min_learn >= min_init and max_learn <= max_init) or
            (min_init >= min_learn and max_init <= max_learn)
        ) else 0

        metrics.append({'IoU': iou, 'Deviation': deviation, 'Inclusion': inclusion})

    mean_iou = np.mean([m['IoU'] for m in metrics])
    mean_deviation = np.mean([m['Deviation'] for m in metrics])
    inclusion_ratio = np.mean([m['Inclusion'] for m in metrics])

    return metrics, mean_iou, mean_deviation, inclusion_ratio