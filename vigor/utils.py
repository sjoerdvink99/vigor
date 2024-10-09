import random
import networkx as nx
from vigor.graph import Graph

def is_spatial(key):
    spatial_keywords = {
        'lat', 'lon', 'latitude', 'longitude', 'location', 'address', 
        'geolocation', 'coord', 'coordinates'
    }
    return any(keyword in key.lower() for keyword in spatial_keywords)


def is_temporal(key):
    temporal_keywords = {
        'time', 'date', 'timestamp', 'datetime', 'year', 'month', 'day', 
        'hour', 'minute', 'second', 'duration', 'epoch'
    }
    return any(keyword in key.lower() for keyword in temporal_keywords)

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