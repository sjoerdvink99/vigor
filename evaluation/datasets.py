import os
import urllib.request
import zipfile

import networkx as nx
import numpy as np
import pandas as pd

from src import Graph


def download_tudataset(name, cache_dir="data/tudatasets"):
    os.makedirs(cache_dir, exist_ok=True)
    dataset_dir = os.path.join(cache_dir, name)
    if os.path.exists(dataset_dir):
        return dataset_dir

    url = f"https://www.chrsmrrs.com/graphkerneldatasets/{name}.zip"
    zip_path = os.path.join(cache_dir, f"{name}.zip")
    print(f"  Downloading {name} from {url}...")
    urllib.request.urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(cache_dir)
    os.remove(zip_path)

    return dataset_dir


def parse_tudataset(name, cache_dir="data/tudatasets"):
    dataset_dir = os.path.join(cache_dir, name)

    with open(os.path.join(dataset_dir, f"{name}_graph_indicator.txt")) as f:
        node_to_graph = [int(line.strip()) for line in f]

    with open(os.path.join(dataset_dir, f"{name}_graph_labels.txt")) as f:
        graph_labels = [int(line.strip()) for line in f]

    n_graphs = len(graph_labels)
    graphs = [nx.Graph() for _ in range(n_graphs)]

    for node_idx, graph_id in enumerate(node_to_graph):
        node_num = node_idx + 1
        graphs[graph_id - 1].add_node(node_num)

    with open(os.path.join(dataset_dir, f"{name}_A.txt")) as f:
        for line in f:
            parts = line.strip().split(",")
            u, v = int(parts[0].strip()), int(parts[1].strip())
            graph_id = node_to_graph[u - 1]
            graphs[graph_id - 1].add_edge(u, v)

    return graphs, graph_labels


def extract_features(graphs, max_nodes=200):
    records = []
    for G_nx in graphs:
        if G_nx.number_of_nodes() == 0 or G_nx.number_of_nodes() > max_nodes:
            continue
        try:
            G = Graph()
            G.from_existing_graph(G_nx)
            stats = G.get_statistics()
            records.append(stats)
        except Exception:
            pass
    return pd.DataFrame(records)


def load_domain_data(datasets, n_per_domain=150, seed=0, cache_dir="data/tudatasets"):
    rng = np.random.default_rng(seed)
    dfs, labels_list = [], []

    for name in datasets:
        print(f"Loading {name}...")
        download_tudataset(name, cache_dir)
        graphs, _ = parse_tudataset(name, cache_dir)
        df = extract_features(graphs, max_nodes=200)

        if len(df) == 0:
            print(f"  Warning: no features extracted for {name}, skipping")
            continue

        n = min(n_per_domain, len(df))
        idx = rng.choice(len(df), size=n, replace=False)
        dfs.append(df.iloc[idx].reset_index(drop=True))
        labels_list.extend([name] * n)
        print(f"  Loaded {n} graphs from {name}")

    if not dfs:
        raise ValueError("No data loaded from any dataset")

    return pd.concat(dfs, ignore_index=True), pd.Series(labels_list)


def one_hot_graph_type(df):
    df = df.copy()
    for i in range(1, 5):
        df[f"graph_type_{i}"] = (df["graph_type"] == i).astype(int)
    return df.drop(columns=["graph_type"])
