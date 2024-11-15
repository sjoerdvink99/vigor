from .visualization_types import VisualizationType

predicates = [
    # Node-Link
    (VisualizationType.NODELINK, 'density', 0, 0.1),
    (VisualizationType.NODELINK, 'is_directed', 0.5, 1),
    (VisualizationType.NODELINK, 'self_loops', 0, 50),
    (VisualizationType.NODELINK, 'components', 1, 5),
    (VisualizationType.NODELINK, 'avg_degree', 1, 3),
    (VisualizationType.NODELINK, 'clustering_coefficient', 0.1, 0.4),
    (VisualizationType.NODELINK, 'node_types', 1, 3),
    (VisualizationType.NODELINK, 'edge_types', 1, 2),
    (VisualizationType.NODELINK, 'eccentricity', 0, 5),
    
    # Matrix
    (VisualizationType.MATRIX, 'density', 0.1, 1),
    (VisualizationType.MATRIX, 'avg_degree', 10, 50),
    (VisualizationType.MATRIX, 'modularity', 0.3, 0.7),
    (VisualizationType.MATRIX, 'betweenness_centrality', 0.2, 0.5),
    (VisualizationType.MATRIX, 'eigenvector_centrality', 0.2, 0.8),
    (VisualizationType.MATRIX, 'node_attributes', 2, 10),
    (VisualizationType.MATRIX, 'edge_attributes', 1, 5),

    # NodeTrix
    (VisualizationType.NODETRIX, 'communities', 4, 10),
    (VisualizationType.NODETRIX, 'clustering_coefficient', 0.5, 1),
    (VisualizationType.NODETRIX, 'density', 0.1, 0.5),
    (VisualizationType.NODETRIX, 'node_types', 2, 5),
    (VisualizationType.NODETRIX, 'modularity', 0.3, 0.8),
    (VisualizationType.NODETRIX, 'avg_degree', 5, 15),
    (VisualizationType.NODETRIX, 'node_attributes', 3, 10),
    (VisualizationType.NODETRIX, 'edge_types', 1, 3),

    # Node-Link Map
    (VisualizationType.NODELINK_MAP, 'is_spatial', 0.5, 1),
    (VisualizationType.NODELINK_MAP, 'is_directed', 0, 1),
    (VisualizationType.NODELINK_MAP, 'avg_degree', 1, 5),
    (VisualizationType.NODELINK_MAP, 'components', 1, 5),
    (VisualizationType.NODELINK_MAP, 'degree_assortativity', -0.5, 0.5),
    (VisualizationType.NODELINK_MAP, 'planar', 0.5, 1),

    # PaohVis
    (VisualizationType.PAOHVIS, 'n_nodes', 50, 500),
    (VisualizationType.PAOHVIS, 'node_types', 3, 6),
    (VisualizationType.PAOHVIS, 'edge_types', 2, 5),
    (VisualizationType.PAOHVIS, 'density', 0.05, 0.2),
    (VisualizationType.PAOHVIS, 'avg_degree', 5, 10),
    (VisualizationType.PAOHVIS, 'transitivity', 0.2, 0.6),

    # Chord Diagram
    (VisualizationType.CHORD_DIAGRAM, 'n_nodes', 0, 6),
    (VisualizationType.CHORD_DIAGRAM, 'edge_types', 1, 3),
    (VisualizationType.CHORD_DIAGRAM, 'clustering_coefficient', 0.3, 0.7),
    (VisualizationType.CHORD_DIAGRAM, 'components', 1, 2),
    (VisualizationType.CHORD_DIAGRAM, 'avg_degree', 2, 4),
    (VisualizationType.CHORD_DIAGRAM, 'parallel_edges', 0, 5),

    # Treemap
    (VisualizationType.TREEMAP, 'graph_type', 0.5, 1.5),
    (VisualizationType.TREEMAP, 'modularity', 0.5, 1),
    (VisualizationType.TREEMAP, 'n_nodes', 50, 200),
    (VisualizationType.TREEMAP, 'node_attributes', 5, 20),
    (VisualizationType.TREEMAP, 'edge_attributes', 0, 2),
    (VisualizationType.TREEMAP, 'components', 1, 1),
    (VisualizationType.TREEMAP, 'is_spatial', 0, 1)
]