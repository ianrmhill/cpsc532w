"""Entry point for the BOED-based analog circuit debug tool."""

from itertools import product
import torch as tc
import pyro
from pyro.optim import Adam
from pyro.contrib.oed.eig import marginal_eig, nmc_eig
import matplotlib.pyplot as plt

__all__ = ['guided_debug']

example_circuit = None


def eval_eigs(prob_mdl, tests, obs_labels=None, circ_prm_labels=None):
    """
    Compute (estimate) the expected information gain of each candidate test given the current probabilistic circuit
    model state.
    """
    eig = nmc_eig(
        prob_mdl,
        tests,           # design, or in this case, tensor of possible designs
        ['Scale-State'], # site label of observations, could be a list
        ['BH'],          # site label of 'targets' (latent variables), could also be list
        N=100,           # number of samples to draw per step in the expectation
        M=100)           # number of gradient steps

    return eig.detach()


def condition_fault_model(fault_mdl, inputs, measured):
    """
    Numerically estimate the posterior probabilities of the various candidate faults within a circuit using Bayes'
    rule conditioned on the observed circuit measurements.

    Currently, we use importance sampling for the numerical estimation technique.
    """
    cond_mdl = pyro.condition(fault_mdl, {'Scale-State': tc.tensor(measured)})

    sampler = pyro.infer.Importance(cond_mdl, num_samples=10000)
    results = sampler.run(inputs)
    normed_w = sampler.get_normalized_weights()
    resamples = tc.distributions.Categorical(normed_w).sample((10000,))
    sampled_vals = {}
    sampled_vals['BH'] = [results.exec_traces[s].nodes['BH']['value'] for s in resamples]

    # Update the current model based on the posterior
    sampled_vals = tc.tensor([results.exec_traces[s].nodes[f"BH"]['value'] for s in resamples])
    for i in range(8):
        ball_priors[i] = tc.count_nonzero(sampled_vals == i) / sampled_vals.size


def guided_debug(circuit=example_circuit, mode='simulated', circ_simulator=None):
    # Setup of general objects needed for the guided debug process
    print(f"Starting guided debug for circuit {circuit} using Debugs Buddy...")

    # Construct possible test voltages to apply
    v_list = tc.linspace(0, 1, 10)
    # With two test voltages we consider every 100mV steps and every possible combination of the two sources
    candidate_tests = tc.tensor(list(product(v_list, repeat=2)), dtype=tc.float)

    # With the circuit to debug defined, we can begin recommending measurements to determine implementation faults
    pyro.clear_param_store()
    while True:
        # First we determine what test inputs to apply to the circuit next
        print(f"Determining next best test to conduct...")
        best_test = int(tc.argmax(eval_eigs(curr_mdl, candidate_tests)).float().detach())

        # Apply the selected test inputs to the circuit and collect measurements
        if mode == 'simulated':
            if not circ_simulator:
                raise Exception('Cannot simulate faulty circuit behaviour without simulation model.')
            print(f"Next best test: {candidate_tests[best_test]}.")
            measured = circ_simulator.run(candidate_tests[best_test])
            print(f"Measured from test: {measured}.")
        else:
            # If in real-world guided debug mode the user must collect the measurements manually
            pass

        # Now we condition the fault model on the measured data
        print(f"Updating probable faults based on measurement data...")
        curr_mdl = condition_fault_model(curr_mdl, candidate_tests[best_test], measured)

        # Now print the probable circuit model for the user to view
        input('Press Enter to run another cycle...')


if __name__ == '__main__':
    guided_debug()