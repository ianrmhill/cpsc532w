# Standard imports
import torch as tc
from time import time
import numpy as np
from sklearn.preprocessing import normalize

# Project imports
from evaluation_based_sampling import evaluate_program
from graph_based_sampling import evaluate_graph
from utils import log_sample


def flatten_sample(sample):
    if type(sample) is list: # NOTE: Nasty hack for the output from program 4 of homework 2
        flat_sample = tc.concat([element.flatten() for element in sample])
    else:
        flat_sample = sample
    return flat_sample


def get_sample(ast_or_graph, mode, verbose=False):
    if mode == 'desugar':
        ret, sig, _ = evaluate_program(ast_or_graph, verbose=verbose)
    elif mode == 'graph':
        ret, sig, _ = evaluate_graph(ast_or_graph, verbose=verbose)
    else:
        raise ValueError('Mode not recognised')
    ret = flatten_sample(ret)
    return ret, sig


def prior_samples(ast_or_graph, mode, num_samples, tmax=None, wandb_name=None, verbose=False):
    '''
    Generate a set of samples from the prior of a FOPPL program
    '''
    samples = []
    if (tmax is not None): max_time = time()+tmax
    for i in range(num_samples):
        sample, _ = get_sample(ast_or_graph, mode, verbose)
        if wandb_name is not None: log_sample(sample, i, wandb_name=wandb_name)
        samples.append(sample)
        if (tmax is not None) and time() > max_time: break
    return samples


def importance_sampling(ast, num_samples, tmax=None, wandb_name=None, verbose=False):
    sigma = {'logw': 0}
    samples = []
    weights = []
    for l in range(num_samples):
        r_l, sigma_l = evaluate_program(ast, sigma)
        samples.append(r_l)
        weights.append(sigma_l['logw'])
    # Now that we have samples and log probabilities for those samples, we need to resample from those samples given the
    # probabilistic weights of getting those samples (using a categorical-style distribution)
    samples = np.array(samples)
    weights = np.array(weights)
    weights = weights / -1
    #weights = np.exp(weights)
    weights = weights / sum(weights)
    resampler = getattr(np.random.default_rng(), 'choice')
    return resampler(size=num_samples, a=samples, p=weights)


def mh_in_gibbs(graph, num_samples, wandb_name=None, verbose=False):
    samples = []
    for l in range(num_samples):
        for v in graph.nodes:
            mh_sample()
            val = () / ()
            accept_prob = min(1, val)
