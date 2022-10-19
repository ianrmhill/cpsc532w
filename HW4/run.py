# Standard imports
import numpy as np
import torch as tc
from time import time
import wandb
import hydra

# Project imports
from daphne import load_program
from var_inf import variational_inference, sample_posterior
from evaluation_based_sampling import abstract_syntax_tree
from graph_based_sampling import Graph
from utils import wandb_plots_homework4


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
        cols = None
        wandb_name = 'Program %s samples'%i if wandb_run else None
        print('Running: '+prog_set+':' ,i)
        print('Maximum samples [log10]:', np.log10(num_samples))
        print('Maximum time [s]:', tmax)
        print('Evaluation scheme:', mode)

        # Load the program, perform VI, then draw posterior samples
        program_graph = load_program(daphne_dir, daphne_prog(i), json_prog(i), mode='desugar', compile=compile)
        program_graph = create_class(program_graph, 'graph')
        posterior = variational_inference(program_graph, wandb_name=wandb_name)
        samples = sample_posterior(posterior, num_samples)

        # Calculate some properties of the data
        try:
            print('Samples shape:', samples.shape)
            print('First sample:', samples[0])
            print('Sample mean:', samples.mean(axis=0))
            print('Sample standard deviation:', samples.std(axis=0))
        except Exception:
            print('Couldn\'t convert samples to tensor form')

        # Weights & biases plots
        if wandb_run: wandb_plots_homework4(samples, i, cols)

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