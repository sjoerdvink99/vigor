import networkx as nx
from typing import Dict, Any, List
from dataclasses import dataclass, field
from networkx.algorithms.community import girvan_newman, modularity
from .utils import is_spatial, is_temporal

@dataclass
class Graph(nx.Graph):
    # Topological Measures
    n_nodes: int = field(init=False)
    n_edges: int = field(init=False)
    density: float = field(init=False)
    n_components: int = field(init=False)
    diameter: int = field(init=False)
    avg_shortest_path_length: float = field(init=False)
    transitivity: float = field(init=False)
    is_bipartite: int = field(init=False)
    is_directed_int: int = field(init=False)
    graph_type: int = field(init=False)
    has_spatial_attributes: int = field(init=False)
    has_temporal_attributes: int = field(init=False)
    eccentricity_avg: float = field(init=False)
    radius: int = field(init=False)
    modularity: float = field(init=False)

    # Node Measures
    node_types: int = field(init=False)
    node_attributes: int = field(init=False)
    avg_degree: float = field(init=False)
    max_degree: int = field(init=False)
    min_degree: int = field(init=False)
    clustering_coefficient: float = field(init=False)
    assortativity: float = field(init=False)
    avg_betweenness_centrality: float = field(init=False)

    # Edge Measures
    edge_types: int = field(init=False)
    edge_attributes: int = field(init=False)
    n_parallel_edges: int = field(init=False)
    n_self_loops: int = field(init=False)

    def __post_init__(self):
        """ Initialize the graph and extract statistics. """
        super().__init__()

    # New method to initialize with another graph
    def from_existing_graph(self, graph: nx.Graph):
        """Initialize this graph from an existing NetworkX graph."""
        self.add_nodes_from(graph.nodes(data=True))
        self.add_edges_from(graph.edges(data=True))

    def calculate_modularity(self) -> float:
        """Calculate modularity based on detected communities."""
        try:
            # Check if the graph has enough nodes and edges
            if self.number_of_edges() == 0 or self.number_of_nodes() == 0:
                raise ValueError("Graph must have at least one edge and one node to calculate modularity.")
            
            # Detect communities using the Girvan-Newman method
            communities_generator = girvan_newman(self)
            top_level_communities = next(communities_generator, None)
            
            # Ensure that communities were detected
            if not top_level_communities:
                raise ValueError("No communities were detected.")
            
            communities = tuple(sorted(c) for c in top_level_communities)

            # Calculate modularity based on the detected communities
            return modularity(self, communities)
        except Exception as e:
            print(f"Error calculating modularity: {e}")
            return 0

    def extract_statistics(self) -> None:
        """Extract statistics from the graph."""
        # Topological Measures
        self.n_nodes = self.number_of_nodes()
        self.n_edges = self.number_of_edges()
        self.is_bipartite = 1 if nx.is_bipartite(self) else 0
        self.is_directed_int = int(self.is_directed())
        self.density = nx.density(self)
        self.n_components = nx.number_weakly_connected_components(self) if self.is_directed() else nx.number_connected_components(self)
        self.transitivity = nx.transitivity(self)

        # If the graph is connected, calculate diameter, radius, and shortest path length
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

        # Graph type
        if nx.is_tree(self):
            self.graph_type = 1  # Tree
        elif self.n_nodes == self.n_edges and all(degree == 2 for _, degree in self.degree()):
            self.graph_type = 2  # Cycle
        else:
            self.graph_type = 3  # General

        # Node Measures
        node_types_set = {data.get('type') for _, data in self.nodes(data=True) if 'type' in data}
        self.node_types = len(node_types_set)
        self.node_attributes = sum(len(data) for _, data in self.nodes(data=True))

        degrees = [degree for _, degree in self.degree()]
        self.avg_degree = sum(degrees) / self.n_nodes if self.n_nodes > 0 else 0
        self.max_degree = max(degrees) if degrees else 0
        self.min_degree = min(degrees) if degrees else 0
        self.clustering_coefficient = nx.average_clustering(self)

        # Betweenness centrality
        betweenness = nx.betweenness_centrality(self)
        self.avg_betweenness_centrality = sum(betweenness.values()) / len(betweenness) if betweenness else 0.0

        # Edge Measures
        edge_types_set = {data.get('type') for _, _, data in self.edges(data=True) if 'type' in data}
        self.edge_types = len(edge_types_set)
        self.edge_attributes = sum(len(data) for _, _, data in self.edges(data=True))

        self.n_self_loops = nx.number_of_selfloops(self)
        self.n_parallel_edges = sum(1 for u, v, k in self.edges(keys=True) if self.number_of_edges(u, v) > 1) if isinstance(self, (nx.MultiGraph, nx.MultiDiGraph)) else 0

        # Assortativity
        if self.n_edges > 0:
            try:
                self.assortativity = nx.degree_assortativity_coefficient(self)
            except Exception as e:
                print(f"Error calculating assortativity: {e}")
                self.assortativity = float('nan')
        else:
            self.assortativity = float('nan')

        # Check for spatial and temporal attributes using utility functions
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

        try:
            self.modularity = self.calculate_modularity()
        except Exception as e:
            print(f"Error calculating modularity: {e}")
            self.modularity = float('nan')

    def get_statistic(self, name: str) -> Any:
        """Retrieve a statistic by its name."""
        self.extract_statistics()
        return getattr(self, name, None)

    def get_statistics(self) -> Dict[str, Any]:
        """Convert the statistics to a dictionary format."""
        self.extract_statistics()
        return {field_name: getattr(self, field_name) for field_name in self.__dataclass_fields__}

    def get_statistics_values(self) -> List[Any]:
        """Get the values of all statistics."""
        return [getattr(self, field_name) for field_name in self.__dataclass_fields__]