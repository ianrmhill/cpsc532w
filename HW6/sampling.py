# Standard imports
import time
from pyrsistent import pmap, m
from multiprocessing import Pool

# Project imports
from evaluator import hoppl_eval, evaluate, standard_env
from utils import log_sample_to_wandb, log_samples_to_wandb
from utils import resample_using_importance_weights, check_addresses


# Have to declare these outside for multiprocessing to work
def output_same(x):
    return x


def get_samples(ast:dict, num_samples:int, tmax=None, inference=None, wandb_name=None, verbose=False):
    '''
    Get some samples from a HOPPL program
    '''
    if inference is None:
        samples = get_prior_samples(ast, num_samples, tmax, wandb_name, verbose)
    elif inference == 'IS':
        samples = get_importance_samples(ast, num_samples, tmax, wandb_name, verbose)
    elif inference == 'SMC':
        samples = get_SMC_samples(ast, num_samples, wandb_name, verbose)
    else:
        print('Inference scheme:', inference, type(inference))
        raise ValueError('Inference scheme not recognised')
    return samples


def get_prior_samples(ast:dict, num_samples:int, tmax=None, wandb_name=None, verbose=False):
    '''
    Generate a set of samples from the prior of a HOPPL program
    '''
    samples = []
    if (tmax is not None): max_time = time.time() + tmax
    for i in range(num_samples):
        sample, _ = evaluate(ast, verbose=verbose)
        if wandb_name is not None: log_sample_to_wandb(sample, i, wandb_name=wandb_name)
        samples.append(sample)
        if (tmax is not None) and (time.time() > max_time): break
    return samples


def get_importance_samples(ast:dict, num_samples:int, tmax=None, wandb_name=None, verbose=False):
    '''
    Generate a set of importance samples from a HOPPL program
    '''
    # NOTE: Fill this in
    return None


def run_till_address(run_state, score_func):
    continuation = run_state['cont']
    sigma = run_state['log_w']
    next_args = run_state['args']
    while True:
        # Run the program to the next return
        if type(continuation) == list:
            stop = hoppl_eval(continuation, sigma, standard_env(), mode='smc')('start', score_func)
        else:
            stop = continuation(*next_args)
        # Process the return from eval
        if type(stop) != tuple:
            return 'return', run_state['id'], 'stop', stop, sigma
        else:
            if stop[0] == 'observe':
                sigma += stop[2].log_prob(stop[3])
                next_args = [stop[3], stop[3]]
                return 'observe', run_state['id'], stop[1], sigma, stop[4], next_args
            else:
                # sigma += stop[2].log_prob(stop[3])
                next_args = [stop[3]]
                continuation = stop[4]


def get_SMC_samples(ast:dict, num_samples:int, wandb_name=None, verbose=False):
    '''
    Generate a set of samples via Sequential Monte Carlo from a HOPPL program
    '''
    run_states = {f"run{id}": m(id=f"run{id}", log_w=0, cont=ast, args=None) for id in range(num_samples)}

    # Use a process pool, 5 is selected based on my desktop CPU having 6 cores, 1 is saved for browsing reddit while
    # SMC programs run
    with Pool(processes=5) as pool:
        while True:
            # Run to the next observe or program return
            run_stops, samples, log_ws = [], [], []
            #for rtrn in pool.starmap(run_till_address, [(state, output_same) for state in run_states.values()]):
            #    run_stops.append(rtrn)
            for run_state in run_states.values():
                run_stops.append(run_till_address(run_state, output_same))

            # First check that all parallel runs have stopped at the same point in the program
            curr_address, exit_type = None, None
            for run in run_stops:
                if not exit_type:
                    exit_type = run[0]
                if not curr_address:
                    curr_address = run[2]
                if run[2] != curr_address:
                    raise Exception('Not all parallel executions stopped at same program location.')

            if exit_type == 'observe':
                # We've hit an observe
                # First update all the run states based on the returned data from the observe statement
                for run in run_stops:
                    run_states[run[1]] = m(id=run[1], log_w=0, cont=run[4], args=run[5])
                # Now perform resampling
                for run in run_stops:
                    samples.append(run[1]) # These are the parallel run ID's!
                    log_ws.append(run[3])
                resample_parents = resample_using_importance_weights(samples, log_ws)
                for i, run_id in enumerate(run_states):
                    run_states[run_id] = run_states[resample_parents[i]].set('id', run_id)
            else:
                # A final sample value has been returned!
                for run in run_stops:
                    samples.append(run[3])
                    log_ws.append(run[4])
                print(len(samples))
                return resample_using_importance_weights(samples, log_ws, wandb_name=wandb_name)
