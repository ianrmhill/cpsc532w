# Standard imports
import numpy as np
import torch as tc
from time import time
import wandb
import hydra

# Project imports
from daphne import load_program
from tests import is_tol, run_probabilistic_test, load_truth
from general_sampling import importance_sampling, mh_in_gibbs
from evaluation_based_sampling import abstract_syntax_tree
from graph_based_sampling import Graph
from utils import wandb_plots_homework3


def create_class(ast_or_graph, mode):
    if mode == 'desugar':
        return abstract_syntax_tree(ast_or_graph)
    elif mode == 'graph':
        return Graph(ast_or_graph)
    else:
        raise ValueError('Input type not recognised')


def run_programs(programs, mode, prog_set, base_dir, daphne_dir, num_samples=int(1e3), tmax=None, compile=False, wandb_run=False, verbose=False,):

    # File paths
    prog_dir = base_dir+'\\programs\\'+prog_set+'\\'
    daphne_prog = lambda i: prog_dir+'%d.daphne'%(i)
    json_prog = lambda i: prog_dir+'%d_%s.json'%(i, mode)
    results_file = lambda i: 'data/%s/%d_%s.dat'%(prog_set, i, mode)

    for i in programs:

        # Draw samples
        t_start = time()
        wandb_name = 'Program %s samples'%i if wandb_run else None
        print('Running: '+prog_set+':' ,i)
        print('Maximum samples [log10]:', np.log10(num_samples))
        print('Maximum time [s]:', tmax)
        print('Evaluation scheme:', mode)

        # Switch based on inference mode
        if mode == 'IS':
            ast_or_graph = load_program(daphne_dir, daphne_prog(i), json_prog(i), mode='desugar', compile=compile)
            ast_or_graph = create_class(ast_or_graph, 'desugar')
            samples = importance_sampling(ast_or_graph, num_samples=num_samples, wandb_name=wandb_name)
        elif mode == 'MHG':
            ast_or_graph = load_program(daphne_dir, daphne_prog(i), json_prog(i), mode='graph', compile=compile)
            ast_or_graph = create_class(ast_or_graph, 'graph')
            samples = mh_in_gibbs(ast_or_graph, num_samples=num_samples, wandb_name=wandb_name)
        else:
            ast_or_graph = load_program(daphne_dir, daphne_prog(i), json_prog(i), mode='graph', compile=compile)
            raise Exception('Not yet implemented!')
        #np.savetxt(results_file(i), samples)

        # Calculate some properties of the data
        #samples = tc.stack(samples).type(tc.float)
        #print('Samples shape:', samples.shape)
        #print('First sample:', samples[0])
        #print('Sample mean:', samples.mean(axis=0))
        #print('Sample standard deviation:', samples.std(axis=0))

        # Weights & biases plots
        if wandb_run: wandb_plots_homework3(samples, i)

        # Finish
        t_finish = time()
        print('Time taken [s]:', t_finish-t_start)
        print('Number of samples:', len(samples))
        print('Finished program {}\n'.format(i))


@hydra.main(version_base=None, config_path='', config_name='config')
def run_all(cfg):

    # Configuration
    wandb_run = cfg['wandb_run']
    mode = cfg['mode']
    num_samples = int(cfg['num_samples'])
    compile = cfg['compile']
    base_dir = cfg['base_dir']
    daphne_dir = cfg['daphne_dir']

    # Initialize W&B
    if wandb_run: wandb.init(project='HW3-'+mode, entity='cs532-2022',
                             settings=wandb.Settings(program_relpath="run.py", disable_git=True, disable_code=True))

    # Programs
    programs = cfg['HW3_programs']
    run_programs(programs, mode=mode, prog_set='HW3', base_dir=base_dir, daphne_dir=daphne_dir, num_samples=num_samples,
        compile=compile, wandb_run=wandb_run, verbose=False)

    # Finalize W&B
    if wandb_run: wandb.finish()

if __name__ == '__main__':
    run_all()