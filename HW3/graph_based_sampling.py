# Standard imports
import torch as tc
from graphlib import TopologicalSorter

# Project imports
from evaluation_based_sampling import interpret, YummyEnv
from primitives import primitives


def swap_keys_vals(edges):
    """Daphne formats edges as parent->children, whereas our topologic sorter needs the form child->parents."""
    new = {}
    for node in edges.keys():
        for child in edges[node]:
            if child in new.keys():
                new[child].append(node)
            else:
                new[child] = [node]
    return new


class Graph:
    def __init__(self, graph_json):
        self.json = graph_json
        # Perform a topological sort to get a valid execution ordering
        if graph_json[1]['A'] == {}:
            self.sorted = graph_json[1]['V']
        else:
            reversed = swap_keys_vals(graph_json[1]['A'])
            self.sorted = tuple(TopologicalSorter(reversed).static_order())
        self.nodes = graph_json[1]['V']
        self.edges = graph_json[1]['A']
        self.links = graph_json[1]['P']
        self.observe = graph_json[1]['Y']


def evaluate_graph(graph: Graph, verbose=False):
    graph_env = YummyEnv(primitives)
    for node in graph.sorted:
        graph_env.add(node, interpret(graph.links[node], graph_env)[0])
    return interpret(graph.json[2], graph_env)
