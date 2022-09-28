# Standard imports
import torch as tc
from graphlib import TopologicalSorter # NOTE: This is useful

# Project imports
#from evaluation_based_sampling import ... NOTE: You can probably reuse some of the evalution-based sampler here
#from primitives import primitives # NOTE: Otherwise you could import and use this again!

class graph:
    def __init__(self, graph_json):
        self.json = graph_json
        # NOTE: You need to write this!


def evaluate_graph(graph, verbose=False):
    # TODO: You need to write this!
    return tc.tensor(7.), None, None # NOTE: This should (artifically) pass deterministic test 1