import pandas as pd
import numpy as np
import random
import networkx as nx
import os
import matplotlib.pyplot as plt
import seaborn as sns
from vigor.graph import Graph

class GraphData:
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.attributes = list(self.kwargs.keys())
        self.dtypes = {attr: self.get_dtype(args) for attr,args in self.kwargs.items()}
        self.numeric = [attr for attr in self.attributes if self.dtypes[attr] in ('float', 'int')]
        self.binary = [attr for attr in self.attributes if self.dtypes=='binary']

    def get_dtype(self, args):
        if type(args)==int:
            if args>2:
                return 'categorical'
            else:
                return 'binary'
        else:
            if type(args[0])==float:
                return 'float'
            else:
                return 'int'
            
    def sample_attr(self, attr, n):
        dtype = self.dtypes[attr]
        if dtype=='categorical':
            return (np.random.choice(range(1, self.kwargs[attr]+1), size=n))
        elif dtype=='binary':
            return np.random.binomial(1, .5, size=n)
        elif dtype=='int':
            return np.random.randint(*self.kwargs[attr], size=n)
        elif dtype=='float':
            return np.random.uniform(*self.kwargs[attr], size=n)
        
    def sample(self, n):
        return pd.DataFrame({attr: self.sample_attr(attr, n) for attr in self.attributes})

def generate_graphs(n_graphs, nodes_min=2, nodes_max=200, file_path=None):
    """
    Function to generate a list of graphs with random number of nodes and edges
    """
    file_exists = False if file_path is None else os.path.exists(file_path)

    graphs = []
    for i in range(n_graphs):
        rand_val = random.random()
        if rand_val < 0.1:
            r = np.random.randint(1, 10)
            h = np.random.randint(1, 10)
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
            if file_path:
                df.to_csv(file_path, mode='a', header=not file_exists, index=False)
                file_exists = True 
        except:
            print(f"Graph {i} failed to extract statistics")

    df = pd.DataFrame(graphs)
    return df

def encode_categorical(df, attribute):
    graph_types = df[attribute].unique()
    df[f'{attribute}_' + pd.Series(graph_types).astype(str)] = (df[attribute].values[None] == graph_types[:,None]).astype(int).T
    df = df.drop(attribute, axis=1)
    return df

def plot_attribute(data, attr):
    fig, ax = plt.subplots()
    dtype = graph_data.dtypes.get(attr,'categorical')
    if dtype in ('float', 'int'):
        legend = []
        for k,v in data.items():
            v[attr].hist(bins=50, alpha=.3, ax=ax)
            legend.append(k)
        ax.legend(legend)
    else:        
        plot_data = pd.concat([v[[attr]].assign(data=k) for k,v in data.items()])
        sns.histplot(
            data=plot_data,
            x="data", hue=attr,
            multiple="fill", stat="proportion",
            discrete=True, shrink=.8, ax=ax
        )
    return fig

graph_data = GraphData(
    graph_type=4,
    is_directed_int=2,
    has_spatial_attributes=2,
    has_temporal_attributes=2,
    is_bipartite=2,
    n_components=(0, 10),
    avg_betweenness_centrality=(0.0, .33),
    avg_closeness_centrality=(0.0, 1.0),
    avg_eigenvector_centrality=(0.0, .7),
    avg_degree=(0, 100),
    std_degree=(0.0, 8.0),
    clustering_coefficient=(0.0, 1.0),
    transitivity=(0.0, 1.0),
    modularity=(-0.5, 1.0),
    communities=(2, 160),
    avg_shortest_path_length=(1.0, 14.0),
    radius=(1, 14),
    diameter=(1, 18),
    assortativity=(-1.0, 1.0),
    vertex_connectivity=(0, 81),
    eccentricity_avg=(0.0, 17.0),
    n_nodes=(2, 20000),
    node_types=(1, 5),
    node_attributes=(0, 15),
    number_of_isolates=(0, 150),
    density=(0.0, 1.0),
    edge_types=(1, 5),
    edge_attributes=(0, 15),
    n_parallel_edges=(0, 5),
    n_self_loops=(0, 5)
)