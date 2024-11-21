import os
import random
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict
from vigor import Graph, Predicate, VIGOR

def generate_graphs(n_graphs, nodes_min=2, nodes_max=200, file_path=None):
    """
    Function to generate a list of graphs with random number of nodes and edges
    """
    file_exists = os.path.exists(file_path)

    graphs = []
    for i in range(n_graphs):
        rand_val = random.random()
        if rand_val < 0.1:
            r = np.random.randint(1, 10)
            h = np.random.randint(1, 20)
            H = nx.balanced_tree(r, h)
        elif rand_val < 0.2:
            H = nx.cycle_graph(np.random.randint(nodes_min, 30))
        else:
            n = np.random.randint(nodes_min, nodes_max)
            p = np.random.uniform(0, 0.5)
            H = nx.fast_gnp_random_graph(n, p)

        G = Graph()
        G.from_existing_graph(H)

        try:
            statistics = G.get_statistics(testing=True)
            print("Generated statistics for graph", i, statistics)
            graphs.append(statistics)
            
            df = pd.DataFrame([statistics])
            df.to_csv(file_path, mode='a', header=not file_exists, index=False)
            file_exists = True 
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
        lambda pred: pred if np.random.random() <= conformance else np.random.choice(unique_labels)
    )
    
    return final_labels

def get_predicates(vigor, X, y, n_iter=1000):
    failed, predicates = vigor.compute_predicate_sequence(
        X.values,
        y[None],
        attribute_names=X.columns,
        n_iter=n_iter,
    )
    
    if failed:
        return predicates
    else:
        p = Predicate(predicates[0])
        p.fit(X)
        return p

def denormalize_predicate(pred, min_vals, max_vals):
    """
    Denormalize the predicate values back to the original scale.
    """
    for attr, range_vals in pred.clauses.items():
        min_val, max_val = range_vals
        original_min = min_vals[attr]
        original_max = max_vals[attr]
        
        # Reverse the normalization
        denorm_min = min_val * (original_max - original_min) + original_min
        denorm_max = max_val * (original_max - original_min) + original_min
        pred.clauses[attr] = [denorm_min, denorm_max]
    
    return pred

def learn_predicates(df, labels, n_iter=1000):
    """
    Function to learn predicates from the data.
    """
    # Remove columns with only one unique value
    df = df.loc[:, df.nunique() > 1]
    epsilon = 1e-8  # Small value to avoid division by zero
    
    # Normalize the data
    min_vals = df.min()
    max_vals = df.max()
    graphs_normalized = (df - min_vals) / (max_vals - min_vals + epsilon)

    vigor = VIGOR()
    pred_list = {}

    for visualization in labels.unique():
        ypos = (labels == visualization).astype(int).values
        yneg = (labels != visualization).astype(int).values
        
        print(f"Learning predicates for {visualization}")

        # Get predicates for positive and negative samples
        pred_pos = get_predicates(vigor, graphs_normalized, ypos, n_iter=n_iter)
        pred_neg = get_predicates(vigor, graphs_normalized, yneg, n_iter=n_iter)

        # Denormalize the predicates
        pred_pos = denormalize_predicate(pred_pos, min_vals, max_vals)
        pred_neg = denormalize_predicate(pred_neg, min_vals, max_vals)
        pred_list[visualization] = (pred_pos, pred_neg)

    return pred_list

def compute_metrics(initial, learned):
    initial_dict = defaultdict(dict)
    for vis_type, stat_name, min_val, max_val in initial:
        capitalized_vis_type = vis_type.value.upper()
        initial_dict[capitalized_vis_type][stat_name] = [min_val, max_val]
    initial_dict = dict(initial_dict)

    visualizations = learned.keys()
    
    evaluation = {}
    for vis in visualizations:
        initial_pred = initial_dict[vis]
        learned_pred = learned[vis][0]
        stats = initial_pred.keys() & learned_pred.clauses.keys()
        
        scores = {}
        for stat in stats:
            init = initial_pred[stat]
            learn = learned_pred.clauses[stat]

            min_init, max_init = init
            min_learn, max_learn = float(learn[0]), float(learn[1])

            intersection = max(0, min(max_init, max_learn) - max(min_init, min_learn))
            union = max(max_init, max_learn) - min(min_init, min_learn)
            iou = intersection / union if union > 0 else 0
            deviation = (abs(min_init - min_learn) + abs(max_init - max_learn)) / 2
            
            inclusion = 1 if (
                (min_learn >= min_init and max_learn <= max_init) or
                (min_init >= min_learn and max_init <= max_learn)
            ) else 0
                        
            scores[stat] = {'iou': iou, 'deviation': deviation, 'inclusion': inclusion}
        evaluation[vis] = scores

    return evaluation