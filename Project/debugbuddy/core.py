"""Entry point for the BOED-based analog circuit debug tool."""

from itertools import product

import torch.cuda
from scipy.stats import norm
import torch as tc
import pyro
from pyro.contrib.oed.eig import marginal_eig, nmc_eig
import matplotlib.pyplot as plt

__all__ = ['guided_debug']

example_circuit = None


def eval_eigs(prob_mdl, tests, obs_labels=None, circ_prm_labels=None, viz_results=False):
    """
    Compute (estimate) the expected information gain of each candidate test given the current probabilistic circuit
    model state.
    """
    eig = nmc_eig(
        prob_mdl,
        tests,           # design, or in this case, tensor of possible designs
        obs_labels,      # site label of observations, could be a list
        circ_prm_labels, # site label of 'targets' (latent variables), could also be list
        N=1000,           # number of samples to draw per step in the expectation
        M=1000)           # number of gradient steps

    if viz_results:
        plt.figure(figsize=(20, 7))
        x_vals = [f"{round(float(test[0]), 1)}, {round(float(test[1]), 1)}" for test in tests]
        plt.plot(x_vals, eig.detach().numpy(), marker='o', linewidth=2)
        plt.xlabel("Possible inputs")
        plt.xticks(rotation=90, fontsize='x-small')
        plt.ylabel("EIG")
        plt.show()

    return eig.detach()


def condition_fault_model(fault_mdl, inputs, measured, prms, edges):
    """
    Numerically estimate the posterior probabilities of the various candidate faults within a circuit using Bayes'
    rule conditioned on the observed circuit measurements.

    Currently, we use importance sampling for the numerical estimation technique.
    """
    # First generate a bunch of samples from the posterior and accumulate the log probability of each output sample
    cond_mdl = pyro.condition(fault_mdl, {'vo': measured})
    sampler = pyro.infer.Importance(cond_mdl, num_samples=1000)
    results = sampler.run(inputs)

    # Now sample from the set of sampled outputs based on the log probabilities, the resample values are trace indices
    normed_w = sampler.get_normalized_weights()
    resamples = tc.distributions.Categorical(normed_w).sample((1000,))

    # Now take the latent values from each trace in the resampled set and average to get the updated set of beliefs
    new_blfs = {}
    for prm in prms:
        prm_name = f"{prm}-r"
        sampled = tc.tensor([results.exec_traces[s].nodes[prm_name]['value'] for s in resamples])
        mu, std = norm.fit(sampled)
        new_blfs[prm] = [mu, std]
    for edge in edges:
        edge_name = str(sorted(tuple(edge)))
        sampled = tc.tensor([results.exec_traces[s].nodes[f"{edge_name}-r"]['value'] for s in resamples])
        new_blfs[edge_name] = tc.count_nonzero(sampled == 0) / sampled.size(0)
    return new_blfs


def guided_debug(circuit=example_circuit, mode='simulated'):
    # Setup of general objects needed for the guided debug process
    print(f"Starting guided debug using Debugs Buddy...")

    # Setup compute device
    pu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(pu)


    # Construct possible test voltages to apply
    v_list = tc.linspace(0, 1, 11)
    # With two test voltages we consider every 100mV steps and every possible combination of the two sources
    candidate_tests = tc.tensor(list(product(v_list, repeat=2)), dtype=tc.float)
    gnds = tc.zeros(121).unsqueeze(-1)
    candidate_tests = tc.cat((candidate_tests, gnds), -1)

    # Define the initial fault model and the graphical nodes that we will be conditioning and observing
    curr_mdl = circuit.gen_fault_mdl()
    obs_lbls = circuit.get_obs_lbls()
    ltnt_lbls = circuit.get_latent_lbls()

    # With the circuit to debug defined, we can begin recommending measurements to determine implementation faults
    pyro.clear_param_store()
    while True:
        # First we determine what test inputs to apply to the circuit next
        print(f"Determining next best test to conduct...")
        best_test = int(tc.argmax(eval_eigs(curr_mdl, candidate_tests, obs_lbls, ltnt_lbls, True)).float().detach())

        # Apply the selected test inputs to the circuit and collect measurements
        if mode == 'simulated':
            print(f"Next best test: {candidate_tests[best_test]}.")
            measured = circuit.simulate_test(candidate_tests[best_test])
            print(f"Measured from test: {measured}.")
        else:
            # If in real-world guided debug mode the user must collect the measurements manually
            measured = None

        # Now we condition the fault model on the measured data
        print(f"Updating probable faults based on measurement data...")
        new_beliefs = condition_fault_model(curr_mdl, candidate_tests[best_test], measured,
                                            circuit.comp_prms, circuit.edges)
        curr_mdl = circuit.gen_fault_mdl(new_beliefs)

        # Now print the probable circuit model for the user to view
        print('Beliefs updated:')
        print(new_beliefs)
        input('Press Enter to run another cycle...')


if __name__ == '__main__':
    guided_debug()