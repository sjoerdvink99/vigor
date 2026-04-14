import networkx as nx
import pytest
from src.graph import Graph


@pytest.fixture
def simple_graph():
    H = nx.path_graph(10)
    G = Graph()
    G.from_existing_graph(H)
    return G


def test_from_existing_graph(simple_graph):
    assert simple_graph.number_of_nodes() == 10
    assert simple_graph.number_of_edges() == 9


def test_get_statistics_keys(simple_graph):
    stats = simple_graph.get_statistics()
    for key in ["n_nodes", "n_edges", "density", "clustering_coefficient", "modularity", "assortativity"]:
        assert key in stats


def test_density_range(simple_graph):
    stats = simple_graph.get_statistics()
    assert 0 <= stats["density"] <= 1


def test_n_nodes(simple_graph):
    stats = simple_graph.get_statistics()
    assert stats["n_nodes"] == 10


def test_n_edges(simple_graph):
    stats = simple_graph.get_statistics()
    assert stats["n_edges"] == 9


def test_graph_type_tree():
    H = nx.balanced_tree(2, 3)
    G = Graph()
    G.from_existing_graph(H)
    stats = G.get_statistics()
    assert stats["graph_type"] == 1


def test_disconnected_graph_sentinels():
    H = nx.Graph()
    H.add_nodes_from([0, 1, 2])
    H.add_edge(0, 1)
    G = Graph()
    G.from_existing_graph(H)
    stats = G.get_statistics()
    assert stats["diameter"] == -1
    assert stats["avg_shortest_path_length"] == -1
