import random
import networkx as nx
from .graph import Graph

def generate_random_graph(nodes=100, density=0.3):
    graph = Graph()
    graph.from_existing_graph(nx.erdos_renyi_graph(n=nodes, p=density, seed=42))

    for node in graph.nodes():
        graph.nodes[node]['type'] = random.choice(['A', 'B'])
        graph.nodes[node]['attribute'] = random.randint(1, 10)

    for edge in graph.edges():
        graph.edges[edge]['type'] = random.choice(['X', 'Y'])
        graph.edges[edge]['weight'] = random.random()

    return graph.get_statistics()