import pandas as pd
import numpy as np
import networkx as nx
from textwrap import dedent
import torch
from torch import nn
from torch import optim
from vigor.graph import Graph

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device", device)

def extract_statistics(self) -> None:
        """Extract statistics from the graph."""
        # Topological Measures
        print('Topological')
        t = time()
        self.n_nodes = self.number_of_nodes()
        self.n_edges = self.number_of_edges()
        self.is_bipartite = 1 if nx.is_bipartite(self) else 0
        self.is_directed_int = int(self.is_directed())
        self.density = nx.density(self)
        self.n_components = nx.number_weakly_connected_components(self) if self.is_directed() else nx.number_connected_components(self)
        self.transitivity = nx.transitivity(self)
        print(time() - t)

        # If the graph is connected, calculate diameter, radius, and shortest path length
        print('Connected')
        t = time()
        if nx.is_connected(self):
            self.diameter = nx.diameter(self)
            self.avg_shortest_path_length = nx.average_shortest_path_length(self)
            ecc = nx.eccentricity(self)
            self.eccentricity_avg = sum(ecc.values()) / len(ecc) if ecc else 0.0
            self.radius = nx.radius(self)
        else:
            self.diameter = -1
            self.avg_shortest_path_length = -1
            self.eccentricity_avg = -1
            self.radius = -1
        print(time() - t)

        # Graph type
        print('Type')
        t = time()
        if nx.is_tree(self):
            self.graph_type = 1  # Tree
        elif self.n_nodes == self.n_edges and all(degree == 2 for _, degree in self.degree()):
            self.graph_type = 2  # Cycle
        else:
            self.graph_type = 3 if self.density <= 0.1 else 4
        print(time() - t)

        # Node Measures
        print('Node')
        t = time()
        node_types_set = {",".join(data['label']) if isinstance(data['label'], list) else data['label'] 
                  for _, data in self.nodes(data=True) if 'label' in data}
        self.node_types = len(node_types_set)
        print(time() - t)

        # Calculate the average number of attributes per node
        print('Attributes')
        t = time()
        total_node_attributes = sum(len(data) for _, data in self.nodes(data=True))
        self.node_attributes = total_node_attributes / self.n_nodes if self.n_nodes > 0 else 0

        degrees = [degree for _, degree in self.degree()]
        self.avg_degree = sum(degrees) / self.n_nodes if self.n_nodes > 0 else 0
        self.std_degree = stdev(degrees) if len(degrees) > 1 else 0
        self.clustering_coefficient = nx.average_clustering(self)
        self.vertex_connectivity = nx.node_connectivity(self)
#         self.s_metric = nx.s_metric(self, False) if nx.is_connected(self) else -1
#         self.sigma = nx.sigma(self) if nx.is_connected(self) else -1
#         self.is_planar = 1 if nx.is_planar(self) else 0
        self.number_of_isolates = nx.number_of_isolates(self)
        print(time() - t)

        # Betweenness centrality
        print('Centrality')
        t = time()
        betweenness = nx.betweenness_centrality(self)
        self.avg_betweenness_centrality = sum(betweenness.values()) / len(betweenness) if betweenness else 0.0
        
        closeness = nx.closeness_centrality(self)
        self.avg_closeness_centrality = sum(closeness.values()) / len(closeness) if closeness else 0.0

        eigenvector = nx.eigenvector_centrality(self)
        self.avg_eigenvector_centrality = sum(eigenvector.values()) / len(eigenvector) if eigenvector else 0.0
        print(time() - t)

        # Edge Measures
        print('Edges')
        t = time()
        edge_types_set = {data.get('type') for _, _, data in self.edges(data=True) if 'type' in data}
        self.edge_types = len(edge_types_set)
        
        # Calculate the average number of attributes per edge
        total_edge_attributes = sum(len(data) for _, _, data in self.edges(data=True))
        self.edge_attributes = total_edge_attributes / self.n_edges if self.n_edges > 0 else 0

        self.n_self_loops = nx.number_of_selfloops(self)
        self.n_parallel_edges = sum(1 for u, v, k in self.edges(keys=True) if self.number_of_edges(u, v) > 1) if isinstance(self, (nx.MultiGraph, nx.MultiDiGraph)) else 0
        print(time() - t)
        
        # Assortativity
        print('Assortativity')
        t = time()
        if self.n_edges > 0:
            try:
                self.assortativity = float(nx.degree_assortativity_coefficient(self))
            except Exception as e:
                print(f"Error calculating assortativity: {e}")
                self.assortativity = float('nan')
        else:
            self.assortativity = float('nan')
        print(time() - t)

        # Check for spatial and temporal attributes using utility functions
        print('Spatial')
        t = time()
        self.has_spatial_attributes = int(any(
            is_spatial(key) for _, data in self.nodes(data=True) for key in data.keys()
        ) or any(
            is_spatial(key) for _, _, data in self.edges(data=True) for key in data.keys()
        ))

        self.has_temporal_attributes = int(any(
            is_temporal(key) for _, data in self.nodes(data=True) for key in data.keys()
        ) or any(
            is_temporal(key) for _, _, data in self.edges(data=True) for key in data.keys()
        ))
        print(time() - t)

        try:
            print('Modularity')
            t = time()
            self.modularity = self.calculate_modularity()
            print(time() - t)
        except Exception as e:
            print(f"Error calculating modularity: {e}")
            self.modularity = float('nan')

def generate_graphs(num_graphs, nmin=2, nmax=200):
    all_graphs = []
    for i in range(num_graphs):
        n = np.random.randint(nmin, nmax)
        p = np.random.uniform(0, 1)
        print(n, p)
        G = nx.fast_gnp_random_graph(n, p)
        H = Graph()
        H.from_existing_graph(G)
        try:
            extract_statistics(H)
            all_graphs.append(H)
        except:
            print('FAILED')
        print()
    df = pd.DataFrame([{k: getattr(d, k) if hasattr(d, k) else np.NaN for k,v in d.__dataclass_fields__.items()} for d in all_graphs])
    return df

def predict(x, a, mu):
    r"""
    UMAP-inspired predict function.
    A bump function centered at $\\mu$ with extent determined by $1/|a|$.

    $$ pred = \frac{1}{1+ \sum_{i=1}^{p} |a_i| * |x_i - \mu_i|^{b}} $$

    Parameters
    ----------
    x - Torch tensor, shape [n_data_points, n_features]
        Input data points
    a - Torch tensor, shape [n_features]
        A parameter for the bounding box extent. 1/a.abs() is the extent of bounding box at prediction=0.5
    mu - Torch tensor, shape [n_features]
        A parameter for the bounding box center
    b - Scalar.
        Hyperparameter for predict function. Power exponent

    Returns
    -------
    pred - Torch tensor of predction for each point in x, shape = [n_data_points, 1]
    """

    b = 4
    pred = 1 / (1 + ((a.abs() * (x - mu).abs()).pow(b)).sum(1))
    return pred

def compute_predicate_sequence(
    x0,
    selected,
    attribute_names=[],
    n_iter=1000,
    device=device,
):
    """
    x0 - numpy array, shape=[n_points, n_feature]. Data points
    selected - boolean array. shape=[brush_index, n_points] of selection
    """

    n_points, n_features = x0.shape
    n_brushes = selected.shape[0]

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
#     print(selection_std)

    # initialize the bounding box center (mu) at the data centroid, +-0.1 at random
    mu_init = selection_centroids
    a_init = 1 / selection_std
    # a = (a_init + 0.1 * (2 * torch.rand(n_brushes, n_features) - 1)).to(device)
    # mu = mu_init + 0.1 * (2 * torch.rand(n_brushes, x.shape[1], device=device) - 1)
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
    train_res = {}
    try:
        for e in range(n_iter):
            loss_per_brush = []
            for t, st in enumerate(selected):  # for each brush, compute loss
                # TODO try subsample:
                # use all selected data
                # randomly sample unselected data with similar size
                pred = predict(x, a[t], mu[t])
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

            # print('bce', loss_per_brush)
            # print('smoothness', smoothness_loss.item())
            # sparsity_loss = 0
            # sparsity_loss = a.abs().mean() * 100
            total_loss = sum(loss_per_brush) + smoothness_loss  # + sparsity_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            if e % max(1, (n_iter // 10)) == 0:
                # print(pred.min().item(), pred.max().item())
                print(f"[{e:>4}] loss {loss.item()}")
                train_res[e] = {'params': (mu, a), 'loss': loss, 'total_loss': total_loss}
    except:
        return train_res
            
    a.detach_()
    mu.detach_()
    # plt.stem(a.abs().numpy()); plt.show()

    qualities = []
    for t, st in enumerate(selected):  # for each brush, compute quality
        pred = predict(x, a[t], mu[t])
        pred = (pred > 0.5).float()
        correct = (pred == label[t]).float().sum().item()
        total = n_points
        accuracy = correct / total
        # 1 meaning points are selected
        tp = ((pred == 1).float() * (label == 1).float()).sum().item()
        fp = ((pred == 1).float() * (label == 0).float()).sum().item()
        fn = ((pred == 0).float() * (label == 1).float()).sum().item()
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 / (1 / precision + 1 / recall) if precision > 0 and recall > 0 else 0
        print(
            dedent(
                f"""
            brush = {t}
            accuracy = {accuracy}
            precision = {precision}
            recall = {recall}
            f1 = {f1}
        """
            )
        )
        qualities.append(
            dict(brush=t, accuracy=accuracy, precision=precision, recall=recall, f1=f1)
        )

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
            #         should_include = r[k] < 1.0 * (x[:,k].max()-x[:,k].min())
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
    parameters = dict(mu=mu, a=a)
    return predicates, qualities, parameters


class Predicate:
    
    def __init__(self, clause_list=None, mask=None, mask_=None, clauses=None):
        self.clause_list = clause_list
        self.clauses = {clause['attribute']: clause['interval'] for clause in clause_list} if clauses is None else clauses
        self.mask_ = mask
        self.mask = mask_
        self.X = None
        self.attrs = tuple(set(self.clauses.keys()))
        
    def __repr__(self):
        return str(self.clauses)
        
    def fit(self, X):
        self.X = X
        self.mask_ = pd.DataFrame({k: (X[k]<v[1]) & (X[k]>v[0]) for k,v in self.clauses.items()})
        self.mask = self.mask_.all(axis=1)
        
    def copy(self):
        predicate = Predicate(None, self.mask, self.mask_, self.clauses)
        if self.X is not None:
            predicate.fit(self.X)
        return predicate
    
    def label(self, X, col, dim=None):
        if dim is not None:
            not_dim_mask = self.mask_.drop(dim, axis=1).all(axis=1)
            return self.mask_.loc[not_dim_mask,dim].astype(int)
        else:
            return self.mask.astype(int)        
    def update_clause(self, dim, a, b):
        clauses = {k:v for k,v in self.clauses.items()}
        clauses[dim] = [a, b]
        predicate = Predicate(None, clauses=clauses)
        if self.X is not None:
            predicate.fit(self.X)
        return predicate
        
    def label_adjacent(self, X, col, dim):
        p1 = self.update_clause(dim, X[dim].min(), self.clauses[dim][0])
        p2 = self.update_clause(dim, self.clauses[dim][1], X[dim].max())
        return p1, p2, p1.label(X, col, dim), p2.label(X, col, dim)

def get_predicates(X, y, n_iter=1000):
    res = compute_predicate_sequence(
        X.values,
        y[None],
        attribute_names=X.columns,
        n_iter=n_iter,
    )
    if type(res)==dict:
        return res
    else:
        predicates, qualities, parameters = res
        p = Predicate(predicates[0])
        p.fit(X)
        return p

def fit_predicates(df, predicates):
    vistype_predicates = {}
    for vistype, attr, minval, maxval in predicates:
        if attr in df.columns:
            predicate = Predicate(clauses={attr: [minval, maxval]})
            predicate.fit(df)
            if vistype.name in vistype_predicates:
                vistype_predicates[vistype.name].append(predicate)
            else:
                vistype_predicates[vistype.name] = [predicate]
    return vistype_predicates

def get_vistype_labels(predicates):
    return pd.DataFrame({p.attrs[0]: p.mask for p in predicates})

def get_labels(vistype_predicates):
    vistype_labels = {k: get_vistype_labels(v) for k,v in vistype_predicates.items()}
    scores = pd.DataFrame({k: v.sum(axis=1) for k,v in vistype_labels.items()})
    labels = scores.idxmax(axis=1)
    return labels, scores, vistype_labels

def get_predicate_labels(df, predicates):
    vistype_predicates = fit_predicates(df, predicates)
    labels, scores, vistype_labels = get_labels(vistype_predicates)
    return {'predicates': vistype_predicates, 'labels': labels}

def get_predicates_label(X_norm, label, labels, n_iter=1000):
    ypos = (labels == label).values
    yneg = ~ypos
    pred_pos = get_predicates(X_norm, ypos, n_iter=n_iter)
    pred_neg = get_predicates(X_norm, yneg, n_iter=n_iter)
    return pred_pos, pred_neg

def get_predicates_labels(df, attrs, labels, label_names=None, n_iter=1000):
    X = df[attrs]
    X_norm = (X - X.min()) / (X.max() - X.min())
    label_names = labels.unique() if label_names is None else label_names
    return {label: get_predicates_label(X_norm, label, labels, n_iter=n_iter) for label in label_names}