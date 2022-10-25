import numpy as np
import torch as tc
import wandb

# Please add the functions below to your pre-existing utils.py file

def softplus(x, beta=1., threshold=20.):
    '''
    Original softplus: S(x) = (1/beta)*ln(1+exp(beta*x))
    '''
    s = tc.where(x<=threshold, tc.log(tc.exp(beta*x)+1.)/beta, x)
    return s


def inverse_softplus(s, beta=1., threshold=20.):
    '''
    Inverse softplus: x(S) = (1/beta)*ln(-1+exp(beta*S))
    '''
    x = tc.where(s<=threshold, tc.log(tc.exp(beta*s)-1.)/beta, s)
    return x


def log_params(variationals: dict, i: int, wandb_name: str) -> None:
    '''
    Log a set of variational-distribution parameters to W&B
    @params
        variationals: list of distributions corresponding to each sample node at each training step
        i: integer corresponding to epoch
        wandb_name: string name of W&B run
    '''
    wandb_name_here = wandb_name+' params'
    samples_dict = {wandb_name_here+'; epoch': i}
    for node, distribution in variationals.items():
        params = [p.clone().detach().numpy() for p in distribution.params()]
        for i, param in enumerate(params):
            samples_dict[wandb_name_here+'; '+node+'; '+str(i)] = param
    wandb.log(samples_dict)


def log_loss(loss, i: int, wandb_name: str) -> None:
    '''
    Log a set of losses corresponding to each node to W&B
    @params
        losses: list of losses corresponding to each sample node at each training step
        i: integer corresponding to epoch
        wandb_name: string name of W&B run
    '''
    wandb_name_here = wandb_name+' loss'
    wandb_dict = {wandb_name_here+'; epoch': i}
    wandb_dict[wandb_name_here+';'] = loss
    wandb.log(wandb_dict)


def wandb_plots_homework4(samples, program):
    '''
    W&B logging of plots for homework 4
    '''
    wandb_log = {}
    if program == 1:
        data = [[j, sample] for j, sample in enumerate(samples)]
        table = wandb.Table(data=data, columns=['sample', 'mu'])
        wandb_log['Program 1'] = wandb.plot.histogram(table, value='mu', title='Program 1; mu')
    elif program == 2:
        data = [[j]+[part for part in sample] for j, sample in enumerate(samples)]
        table = wandb.Table(data=data, columns=['sample', 'slope', 'bias', 'posterior-predictive'])
        for thing in ['slope', 'bias', 'posterior-predictive']:
            wandb_log['Program 2; '+thing] = wandb.plot.histogram(table, value=thing, title='Program 2; '+thing)
        wandb_log['Program 2; scatter'] = wandb.plot.scatter(table, x='slope', y='bias', title='Program 2; slope vs. bias')
    elif program == 3:
        data = [[j, sample] for j, sample in enumerate(samples)]
        table = wandb.Table(data=data, columns=['sample', 'x'])
        wandb_log['Program 3'] = wandb.plot.histogram(table, value='x', title='Program 3; Are the points from the same cluster?')
    elif program == 4:
        data = np.array(samples)
        xs = np.linspace(0, data.shape[1]-1, num=data.shape[1]) # [0, 1, ..., 129]
        x = []; y = []
        for i in range(data.shape[0]): # Number of samples
            for j in range(data.shape[1]): # 130 values
                x.append(xs[j])
                y.append(data[i, j])
        xedges = np.linspace(-0.5, data.shape[1]-0.5, data.shape[1]+1) # -0.5, 0.5, ..., 129.5
        yedges = np.linspace(-10.5, 10.5, 20+2) # -10, -9, ..., 10
        matrix, _, _ = np.histogram2d(x, y, bins=(xedges, yedges))
        xlabels = xedges[:-1]+0.5; ylabels = yedges[:-1]+0.5
        wandb_log['Program 4; heatmap'] = wandb.plots.HeatMap(xlabels, ylabels, matrix.T)
    elif program == 5:
        data = [[j, sample] for j, sample in enumerate(samples)]
        table = wandb.Table(data=data, columns=['sample', 's'])
        wandb_log['Program 5'] = wandb.plot.histogram(table, value='s', title='Program 5; s')
    else:
        raise ValueError('Program not recognised')
    wandb.log(wandb_log)
