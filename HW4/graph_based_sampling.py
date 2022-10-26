# Standard imports
import torch as tc
import re
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
        # Separate out the sample and observe nodes
        self.sample_nodes = []
        self.observe_nodes = []
        for node in self.nodes:
            if 'sample' in node:
                self.sample_nodes.append(node)
            else:
                self.observe_nodes.append(node)
        self.edges = graph_json[1]['A']
        self.links = graph_json[1]['P']
        self.observe = graph_json[1]['Y']
        self.output_expression = graph_json[2]


def evaluate_graph(graph: Graph, env=None, verbose=False):
    if not env:
        env = YummyEnv(primitives)
        for node in graph.sorted:
            env.add(node, interpret(graph.links[node], 0, env)[0])
    return interpret(graph.json[2], 0, env)


def eval_graph_given_samples(graph: Graph, order, sample_vals, hardfix):
    env = YummyEnv(primitives)
    offset = 0
    for i, node in enumerate(order):
        ia = i + offset
        if node in hardfix:
            env.add(node, sample_vals[ia:ia + 3])
            offset += 2
        else:
            env.add(node, sample_vals[ia])
    for node in graph.sorted:
        if 'sample' in node:
            continue
        elif 'observe' in node:
            env.add(node, interpret(graph.links[node], {'logw': 0}, env)[0])
    return interpret(graph.output_expression, {'logw': 0}, env)[0]


def sample_prior_val(link, sample_vals):
    graph_env = YummyEnv(primitives | sample_vals)
    link[0] = 'sample'
    return interpret(link, {'logw': 0}, graph_env)[0]


def eval_link_prob(node, link, sample_vals):
    graph_env = YummyEnv(primitives | sample_vals)
    # This method evaluates the log probability of a variable outcome given all x and y in a model
    if node in sample_vals:
        graph_env.add('*sample_val*', sample_vals[node])
    log_p = interpret(link, {'logw': 0}, graph_env, mode='p')[1]['logw']
    return log_p # Returning a new tensor was what was causing the autograd failure!!! Baffling


def eval_graph_prob(nodes, links, order, vals, hardfix=None):
    prob = tc.tensor(0, dtype=tc.float)
    env_vals = {}
    offset = 0
    for i, node in enumerate(order):
        ia = i + offset
        if node in hardfix:
            env_vals[node] = vals[ia:ia+3]
            offset += 2
        else:
            env_vals[node] = vals[ia]
    for node in nodes:
        prob += eval_link_prob(node, links[node], env_vals)
    return prob
