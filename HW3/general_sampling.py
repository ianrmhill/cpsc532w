# Standard imports
import torch as tc
from time import time
import numpy as np

# Project imports
from evaluation_based_sampling import evaluate_program
from graph_based_sampling import evaluate_graph, eval_link_prob, sample_prior_val
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


def importance_sampling(ast, num_samples, wandb_name=None, verbose=False):
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


def acceptance(graph, x, x_new, x_old):
    p_new_given_old = eval_link_prob(x, graph.links[x], x_old)
    p_old_given_new = eval_link_prob(x, graph.links[x], x_new)
    # Could compute which variables depend on the current node x, but just be lazy and compute all then the independent
    # probabilities will cancel out in our overall expression
    p_old = 0
    p_new = 0
    for node in graph.nodes:
        try:
            p_old += eval_link_prob(node, graph.links[node], x_old)
            p_new += eval_link_prob(node, graph.links[node], x_new)
        except ValueError:
            # This block is used to capture samples from our random walk that go out of a variable's support
            # e.g. a negative choice of sigma in a normal distribution
            return 0
    # Sum/diff all the log probabilities then convert to our acceptance probability
    log_a = (p_old_given_new + p_new) - (p_new_given_old + p_old)
    return min(1, np.exp(log_a))


def mh_in_gibbs(graph, num_samples, wandb_name=None, verbose=False):
    # Construct our initial sample value
    in_flux = {}
    for node in graph.sorted:
        if node in graph.sample_nodes:
            in_flux[node] = sample_prior_val(graph.links[node], in_flux)
    key_order = list(in_flux.keys())

    # Now begin building samples
    samples = []
    for l in range(1, num_samples+1):
        # Using DUGS variable iteration
        for node in graph.sample_nodes:
            new = in_flux.copy()
            # I'm going with random walk MH here, sigma is our random walk step size parameter
            # Sample from a normal centered around the previous sampled value of the variable to propose a new value
            new[node] = sample_prior_val(graph.links[node], new)
            # Calculate the probability of accepting the sampled value
            accept_prob = acceptance(graph, node, new, in_flux)
            # If accepted, update the sample we're building to use the new sampled value
            if tc.distributions.Uniform(0, 1).sample() < accept_prob:
                in_flux[node] = new[node]
        samples.append(list(in_flux.values()))
    # Remove initial guess since it's not really a true sample
    return samples, key_order


def graph_grad(graph, pos):
    d_x = tc.zeros(len(graph.sample_nodes))
    for node in pos:
        pos[node].requires_grad = True
        pos[node].retain_grad()
    for x, node in enumerate(graph.sample_nodes):
        eval_link_prob(node, graph.links[node], pos).backward()
        d_x[x] = pos[node].grad
    return d_x


def leapfrog(start_pos, start_mom, mass, graph, t_sim, t_step):
    pos, mom = start_pos, start_mom
    # Leapfrog step until we run out of simulation time
    t = 0
    d_pos = graph_grad(graph, start_pos)
    while t < t_sim:
        # Halfway update our momentum based on the position/probability/potential gradient
        mom_int = mom + (1/2 * d_pos * t_step)
        # Update the mass position based on the current velocity and time step
        pos = pos + ((mom_int / mass) * t_step)
        # Calculate the gradient of the probability/potential surface at the new position
        d_pos = graph_grad(graph, pos)
        # Update the momentum to the current position
        mom = mom_int + (1/2 * d_pos * t_step)
        t += t_step
    return pos, mom


def total_energy(pos, mom, mass, graph):
    # Take multidimensional dot product of momentum to get kinetic energy
    kinetic = 1/2 * tc.dot(mom, mom) / mass
    # Now calculate the potential energy by summing the log probabilities of each graph node
    potential = 0
    for node in graph.nodes:
        potential += eval_link_prob(node, graph.links[node], pos)
    # Take inverse since we want to minimize energy, higher prob needs to be lower energy
    potential *= -1
    # Total energy is just the sum of kinetic and potential
    return kinetic + potential


def hamiltonian_montecarlo(graph, num_samples):
    sim_time_per_leap = 10
    time_quantization = 0.1
    kick_strength = 1
    sim_mass = 1
    num_dims = len(graph.sample_nodes)
    # Construct our initial sample value based on the priors
    curr_pos = {}
    for node in graph.sorted:
        if node in graph.sample_nodes:
            curr_pos[node] = sample_prior_val(graph.links[node], curr_pos)
    curr_nrg = 0
    samples = []

    for l in range(1, num_samples+1):
        # Perform a normally-distributed 'kick' to give the simulated mass some momentum
        mom_kick = tc.distributions.Normal(0, kick_strength).sample((num_dims,))
        # Leapfrog integrate some number of steps to simulate the mass moving across the probability surface
        prop_pos, prop_mom = leapfrog(curr_pos, mom_kick, sim_mass, graph, sim_time_per_leap, time_quantization)
        # Calculate the energy of the mass at the new position
        prop_nrg = total_energy(prop_pos, prop_mom, sim_mass, graph)
        # Calculate the acceptance probability and potentially accept the new position in the variable space
        accept_prob = tc.min(tc.tensor(1), tc.exp(curr_nrg - prop_nrg))
        # If the proposed new position is accepted we sample the new position, otherwise return to the old
        if tc.distributions.Uniform(0, 1).sample() < accept_prob:
            curr_pos, curr_nrg = prop_pos, prop_nrg
            samples.append(prop_pos)
        else:
            samples.append(curr_pos)
    return samples
