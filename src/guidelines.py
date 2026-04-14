from .types import VisualizationType, NobreVisualizations

predicates = [
    (VisualizationType.NODELINK, 'density', 0, 0.1),
    (VisualizationType.NODELINK, 'n_components', 1, 3),
    (VisualizationType.NODELINK, 'clustering_coefficient', 0.1, 0.4),
    (VisualizationType.NODELINK, 'node_types', 1, 3),
    (VisualizationType.NODELINK, 'edge_types', 1, 2),
    (VisualizationType.NODELINK, 'eccentricity_avg', 0, 5),

    (VisualizationType.MATRIX, 'density', 0.4, 1),
    (VisualizationType.MATRIX, 'avg_degree', 10, 50),
    (VisualizationType.MATRIX, 'modularity', 0.3, 0.7),
    (VisualizationType.MATRIX, 'avg_betweenness_centrality', 0.2, 0.5),
    (VisualizationType.MATRIX, 'avg_eigenvector_centrality', 0.2, 0.8),
    (VisualizationType.MATRIX, 'node_attributes', 2, 10),
    (VisualizationType.MATRIX, 'edge_attributes', 1, 5),

    (VisualizationType.NODETRIX, 'communities', 4, 10),
    (VisualizationType.NODETRIX, 'clustering_coefficient', 0.5, 1),
    (VisualizationType.NODETRIX, 'density', 0.1, 0.5),
    (VisualizationType.NODETRIX, 'node_types', 2, 5),
    (VisualizationType.NODETRIX, 'modularity', 0.3, 0.8),
    (VisualizationType.NODETRIX, 'avg_degree', 5, 15),
    (VisualizationType.NODETRIX, 'node_attributes', 3, 10),
    (VisualizationType.NODETRIX, 'edge_types', 1, 3),

    (VisualizationType.NODELINK_MAP, 'has_spatial_attributes', 0.5, 1),
    (VisualizationType.NODELINK_MAP, 'avg_degree', 1, 5),
    (VisualizationType.NODELINK_MAP, 'n_components', 1, 5),
    (VisualizationType.NODELINK_MAP, 'assortativity', -0.5, 0.5),

    (VisualizationType.PAOHVIS, 'n_nodes', 50, 100),
    (VisualizationType.PAOHVIS, 'node_types', 3, 6),
    (VisualizationType.PAOHVIS, 'edge_types', 2, 5),
    (VisualizationType.PAOHVIS, 'density', 0.05, 0.2),
    (VisualizationType.PAOHVIS, 'avg_degree', 5, 10),
    (VisualizationType.PAOHVIS, 'transitivity', 0.2, 0.6),

    (VisualizationType.CHORD_DIAGRAM, 'n_nodes', 6, 30),
    (VisualizationType.CHORD_DIAGRAM, 'edge_types', 1, 3),
    (VisualizationType.CHORD_DIAGRAM, 'clustering_coefficient', 0.3, 0.7),
    (VisualizationType.CHORD_DIAGRAM, 'n_components', 1, 3),
    (VisualizationType.CHORD_DIAGRAM, 'density', 0.15, 0.5),

    (VisualizationType.TREEMAP, 'graph_type_1', 0.5, 1),
    (VisualizationType.TREEMAP, 'modularity', 0.5, 1),
    (VisualizationType.TREEMAP, 'n_nodes', 30, 100),
    (VisualizationType.TREEMAP, 'node_attributes', 5, 15),
    (VisualizationType.TREEMAP, 'edge_attributes', 0, 3),
    (VisualizationType.TREEMAP, 'n_components', 1, 2),
]

nobre_predicates = [
    (NobreVisualizations.NODELINK, 'n_nodes', 5, 80),
    (NobreVisualizations.NODELINK, 'density', 0.02, 0.25),
    (NobreVisualizations.NODELINK, 'node_attributes', 0, 3),
    (NobreVisualizations.NODELINK, 'node_types', 1, 2),
    (NobreVisualizations.NODELINK, 'edge_types', 1, 2),

    (NobreVisualizations.NODELINK_POSITIONING, 'n_nodes', 5, 100),
    (NobreVisualizations.NODELINK_POSITIONING, 'density', 0.02, 0.3),
    (NobreVisualizations.NODELINK_POSITIONING, 'has_spatial_attributes', 0.5, 1),
    (NobreVisualizations.NODELINK_POSITIONING, 'node_attributes', 2, 10),
    (NobreVisualizations.NODELINK_POSITIONING, 'node_types', 1, 5),

    (NobreVisualizations.NODELINK_FACETING, 'graph_type_1', 0.5, 1),
    (NobreVisualizations.NODELINK_FACETING, 'n_nodes', 5, 80),
    (NobreVisualizations.NODELINK_FACETING, 'node_attributes', 2, 10),
    (NobreVisualizations.NODELINK_FACETING, 'node_types', 1, 4),

    (NobreVisualizations.MATRIX, 'n_nodes', 20, 100),
    (NobreVisualizations.MATRIX, 'density', 0.3, 1.0),
    (NobreVisualizations.MATRIX, 'node_attributes', 3, 15),
    (NobreVisualizations.MATRIX, 'node_types', 1, 3),
    (NobreVisualizations.MATRIX, 'edge_attributes', 0, 5),
    (NobreVisualizations.MATRIX, 'edge_types', 1, 3),

    (NobreVisualizations.QUILTS, 'n_nodes', 10, 100),
    (NobreVisualizations.QUILTS, 'density', 0.02, 0.4),
    (NobreVisualizations.QUILTS, 'node_types', 3, 6),
    (NobreVisualizations.QUILTS, 'edge_types', 2, 5),
    (NobreVisualizations.QUILTS, 'node_attributes', 5, 15),
    (NobreVisualizations.QUILTS, 'edge_attributes', 3, 10),

    (NobreVisualizations.BIOFABRIC, 'n_nodes', 40, 100),
    (NobreVisualizations.BIOFABRIC, 'density', 0.01, 0.08),
    (NobreVisualizations.BIOFABRIC, 'avg_degree', 1, 8),
    (NobreVisualizations.BIOFABRIC, 'node_attributes', 0, 8),
    (NobreVisualizations.BIOFABRIC, 'node_types', 1, 5),
    (NobreVisualizations.BIOFABRIC, 'edge_attributes', 0, 5),
    (NobreVisualizations.BIOFABRIC, 'edge_types', 1, 4),

    (NobreVisualizations.TREEMAP, 'graph_type_1', 0.5, 1),
    (NobreVisualizations.TREEMAP, 'n_nodes', 30, 100),
    (NobreVisualizations.TREEMAP, 'node_attributes', 5, 15),
    (NobreVisualizations.TREEMAP, 'node_types', 1, 3),

    (NobreVisualizations.SUNBURST, 'graph_type_1', 0.5, 1),
    (NobreVisualizations.SUNBURST, 'n_nodes', 5, 50),
    (NobreVisualizations.SUNBURST, 'node_attributes', 1, 8),
    (NobreVisualizations.SUNBURST, 'node_types', 1, 3),
]
