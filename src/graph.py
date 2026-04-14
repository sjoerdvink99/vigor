import networkx as nx
from typing import Dict, Any
from dataclasses import dataclass, field
from networkx.algorithms.community import greedy_modularity_communities, modularity
from networkx.algorithms.approximation import node_connectivity as approx_node_connectivity
from .utils import is_spatial, is_temporal
from statistics import stdev


@dataclass
class Graph(nx.Graph):
    graph_type: int = field(init=False)
    is_directed_int: int = field(init=False)
    has_spatial_attributes: int = field(init=False)
    has_temporal_attributes: int = field(init=False)
    is_bipartite: int = field(init=False)

    n_components: int = field(init=False)
    avg_betweenness_centrality: float = field(init=False)
    avg_closeness_centrality: float = field(init=False)
    avg_eigenvector_centrality: float = field(init=False)
    avg_degree: float = field(init=False)
    std_degree: float = field(init=False)
    clustering_coefficient: float = field(init=False)
    transitivity: float = field(init=False)
    modularity: float = field(init=False)
    communities: int = field(init=False)
    avg_shortest_path_length: float = field(init=False)
    radius: int = field(init=False)
    diameter: int = field(init=False)
    assortativity: float = field(init=False)
    vertex_connectivity: float = field(init=False)
    eccentricity_avg: float = field(init=False)

    n_nodes: int = field(init=False)
    n_edges: int = field(init=False)
    node_types: int = field(init=False)
    node_attributes: float = field(init=False)
    number_of_isolates: int = field(init=False)
    density: float = field(init=False)
    edge_types: int = field(init=False)
    edge_attributes: float = field(init=False)
    n_parallel_edges: int = field(init=False)
    n_self_loops: int = field(init=False)

    def __post_init__(self):
        super().__init__()

    def from_existing_graph(self, graph: nx.Graph):
        self._source_directed = isinstance(graph, nx.DiGraph)
        self.add_nodes_from(graph.nodes(data=True))
        self.add_edges_from(graph.edges(data=True))

    def _calculate_modularity(self) -> float:
        if self.number_of_edges() == 0 or self.number_of_nodes() == 0:
            self.communities = 0
            return 0.0
        try:
            comms = list(greedy_modularity_communities(self))
            self.communities = len(comms)
            return float(modularity(self, comms))
        except Exception:
            self.communities = 0
            return 0.0

    def extract_statistics(self) -> None:
        self.n_nodes = self.number_of_nodes()
        self.n_edges = self.number_of_edges()

        self.is_bipartite = 1 if nx.is_bipartite(self) else 0
        self.is_directed_int = int(getattr(self, "_source_directed", self.is_directed()))
        self.n_components = (
            nx.number_weakly_connected_components(self)
            if self.is_directed()
            else nx.number_connected_components(self)
        )

        self.density = nx.density(self)
        self.transitivity = nx.transitivity(self)

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

        if nx.is_tree(self):
            self.graph_type = 1
        elif self.n_nodes == self.n_edges and all(degree == 2 for _, degree in self.degree()):
            self.graph_type = 2
        else:
            self.graph_type = 3 if self.density <= 0.1 else 4

        node_types_set = {
            ",".join(data["label"]) if isinstance(data["label"], list) else data["label"]
            for _, data in self.nodes(data=True)
            if "label" in data
        }
        self.node_types = len(node_types_set)
        total_node_attributes = sum(len(data) for _, data in self.nodes(data=True))
        self.node_attributes = total_node_attributes / self.n_nodes if self.n_nodes > 0 else 0.0

        degrees = [degree for _, degree in self.degree()]
        self.avg_degree = sum(degrees) / self.n_nodes if self.n_nodes > 0 else 0.0
        self.std_degree = stdev(degrees) if len(degrees) > 1 else 0.0
        self.clustering_coefficient = nx.average_clustering(self)
        self.vertex_connectivity = approx_node_connectivity(self)
        self.number_of_isolates = nx.number_of_isolates(self)

        betweenness = nx.betweenness_centrality(self)
        self.avg_betweenness_centrality = sum(betweenness.values()) / len(betweenness) if betweenness else 0.0

        closeness = nx.closeness_centrality(self)
        self.avg_closeness_centrality = sum(closeness.values()) / len(closeness) if closeness else 0.0

        try:
            eigenvector = nx.eigenvector_centrality(self, max_iter=200)
        except nx.PowerIterationFailedConvergence:
            eigenvector = {n: 0.0 for n in self.nodes()}
        self.avg_eigenvector_centrality = sum(eigenvector.values()) / len(eigenvector) if eigenvector else 0.0

        edge_types_set = {data.get("type") for _, _, data in self.edges(data=True) if "type" in data}
        self.edge_types = len(edge_types_set)
        total_edge_attributes = sum(len(data) for _, _, data in self.edges(data=True))
        self.edge_attributes = total_edge_attributes / self.n_edges if self.n_edges > 0 else 0.0

        self.n_self_loops = nx.number_of_selfloops(self)
        self.n_parallel_edges = (
            sum(1 for u, v, k in self.edges(keys=True) if self.number_of_edges(u, v) > 1)
            if isinstance(self, (nx.MultiGraph, nx.MultiDiGraph))
            else 0
        )

        if self.n_edges > 0:
            try:
                self.assortativity = float(nx.degree_assortativity_coefficient(self))
            except Exception:
                self.assortativity = -1.0
        else:
            self.assortativity = -1.0

        self.has_spatial_attributes = int(
            any(is_spatial(key) for _, data in self.nodes(data=True) for key in data.keys())
            or any(is_spatial(key) for _, _, data in self.edges(data=True) for key in data.keys())
        )
        self.has_temporal_attributes = int(
            any(is_temporal(key) for _, data in self.nodes(data=True) for key in data.keys())
            or any(is_temporal(key) for _, _, data in self.edges(data=True) for key in data.keys())
        )

        self.modularity = self._calculate_modularity()

    def get_statistics(self) -> Dict[str, Any]:
        self.extract_statistics()
        return {field_name: getattr(self, field_name) for field_name in self.__dataclass_fields__}
