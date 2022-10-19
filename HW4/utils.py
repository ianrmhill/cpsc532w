import wandb


def log_sample(sample, i, wandb_name):
    '''
    Log an individual sample to W&B
    '''
    if len(sample) == 1:
        samples_dict = {wandb_name+'; epoch': i, wandb_name: sample}
    else:
        samples_dict = {wandb_name+'; epoch': i}
        for i, element in enumerate(sample):
            samples_dict[wandb_name+'; '+str(i)] = element
    wandb.log(samples_dict)


def wandb_plots_homework4(samples, program, key_order=None):

    # W&B logging of actual plots
    wandb_log = {}
    if program == 1:
        cols = ['sample', 'mu']
        data = [[j, sample] for j, sample in enumerate(samples)]
        table = wandb.Table(data=data, columns=cols)
        wandb_log['Program 1'] = wandb.plot.histogram(table, value='mu', title='Program 1; mu')
    elif program == 2:
        #if key_order:
        #    for i, name in enumerate(key_order):
        #        if name == 'sample1':
        #            key_order[i] = 'slope'
        #        elif name == 'sample2':
        #            key_order[i] = 'bias'
        #    key_order.insert(0, 'sample')
        #    cols = key_order
        #else:
        cols = ['sample', 'bias', 'slope']
        data = [[j]+[part for part in sample] for j, sample in enumerate(samples)]
        table = wandb.Table(data=data, columns=cols)
        wandb_log['Program 2; slope'] = wandb.plot.histogram(table, value='slope', title='Program 2; slope')
        wandb_log['Program 2; bias'] = wandb.plot.histogram(table, value='bias', title='Program 2; bias')
        wandb_log['Program 2; scatter'] = wandb.plot.scatter(table, x='slope', y='bias', title='Program 2; slope vs. bias')
    elif program == 3:
        cols = ['sample', 'x']
        data = [[j, sample] for j, sample in enumerate(samples)]
        table = wandb.Table(data=data, columns=cols)
        wandb_log['Program 3'] = wandb.plot.histogram(table, value='x', title='Program 3; Are the points from the same cluster?')
    elif program == 4:
        cols = ['sample', 'x']
        data = [[j, sample] for j, sample in enumerate(samples)]
        table = wandb.Table(data=data, columns=cols)
        wandb_log['Program 4'] = wandb.plot.histogram(table, value='x', title='Program 4; Is it raining?')
    elif program == 5:
        cols = ['sample', 'x', 'y']
        data = [[j]+[part for part in sample] for j, sample in enumerate(samples)]
        table = wandb.Table(data=data, columns=cols)
        wandb_log['Program 5; x'] = wandb.plot.histogram(table, value='x', title='Program 5; x')
        wandb_log['Program 5; y'] = wandb.plot.histogram(table, value='y', title='Program 5; y')
        wandb_log['Program 5; scatter'] = wandb.plot.scatter(table, x='x', y='y', title='Program 5; x vs. y')
    else:
        raise ValueError('Program not recognised')
    wandb.log(wandb_log)