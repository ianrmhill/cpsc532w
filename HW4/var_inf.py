import torch as tc

import distributions as dist
import utils
from graph_based_sampling import eval_graph_prob


class Guide:
    def __init__(self, graph, prog_name):
        #q = None
        #for node in graph.sorted:
        # Hard coded guides seems lazy, but actually makes sense since the probabilistic programmer wants control over
        # the form of the guide and such. I'm the one that's lazy.
        match prog_name:
            case 'Program 1':
                self.dists = {'sample2': dist.Normal(tc.tensor(1.), tc.sqrt(tc.tensor(5.)))}
            case 'Program 2':
                self.dists = {'sample1': dist.Normal(tc.tensor(0.), tc.tensor(10.)),
                              'sample2': dist.Normal(tc.tensor(0.), tc.tensor(10.))}
            case 'Program 3':
                self.dists = {f"sample{i}": dist.Normal(tc.tensor(0.), tc.tensor(10.)) for i in range(0, 5, 2)}
                self.dists = self.dists | {f"sample{i}": dist.Gamma(tc.tensor(1.), tc.tensor(1.)) for i in range(1, 6, 2)}
                self.dists['sample6'] = dist.Dirichlet([tc.tensor(1.), tc.tensor(1.), tc.tensor(1.)])
            case 'Program 4':
                self.dists = {f"sample{i}": dist.Normal(tc.tensor(0.), tc.tensor(1.)) for i in range(142)}
            case 'Program 5':
                self.dists = {'sample1': dist.Normal(tc.tensor(0.), tc.tensor(5.))}
                self.evals = {'sample2': tc.distributions.Uniform(0.01, tc.abs('sample1'))}
            case _:
                raise Exception()
        self.node_order = [node for node in self.dists] + [node for node in self.evals]
        self.num_nodes = len(self.node_order)

    def sample(self, num_samples):
        samples = tc.zeros((self.num_nodes, num_samples))
        for i, node in enumerate(self.node_order):
            samples[i] = self.dists[node].sample(sample_shape=(num_samples,))
        return samples

    def log_prob(self, samples, num_samples):
        logps = tc.zeros((num_samples,))
        for i, node in enumerate(self.dists):
            logps += self.dists[node].log_prob(samples[i])
        return logps

    def optim_prms(self):
        prms = []
        for i, node in enumerate(self.dists):
            prms.extend(self.dists[node].optim_params())
        return prms


def variational_inference(p_graph, guide, wandb_name, wandb_run):
    # Initialize things
    num_epochs = 10
    samples_per_epoch = 10
    lr = 0.1
    rho = -1
    learner = tc.optim.Adam(guide.optim_prms(), lr=lr)

    i = 0
    while i < num_epochs:
        # Sample Q
        z = guide.sample(samples_per_epoch)

        # Get the probability of sampling the sampled values
        sample_logp = guide.log_prob(z, samples_per_epoch)

        # Get the probability of full set of sampled values and observations from posterior
        graph_logp = tc.zeros((samples_per_epoch,))
        for s in range(samples_per_epoch):
            graph_logp[s] = eval_graph_prob(p_graph.sorted, p_graph.links, guide.node_order, z[:, s])

        # Calculate gradient of probability expression w.r.t variational parameters
            # Must mean across the right dimension, one dim is parameters, the other is samples
        is_this_loss = rho * (sample_logp * (graph_logp - sample_logp).detach()).mean()
        is_this_loss.backward()

        # Update the guide parameters based on the calculated gradient
        learner.step()
        # Reset gradient for next epoch
        learner.zero_grad()

        # Log stats and increment our epoch
        if wandb_run:
            utils.log_loss(is_this_loss, i, wandb_name)
            utils.log_params(guide.dists, i, wandb_name)
        i += 1

    # Return the trained guide, I've made it mutable
    return guide


def sample_posterior(posterior_guide, num_samples):
    """Note that this is only used for Program 2 for some reason, so I just made it specific instead of general."""
    samples = posterior_guide.sample(num_samples)
    predictive = tc.zeros((1, num_samples))
    for i in range(num_samples):
        predictive[0][i] = (samples[0][i] * 0) + samples[1][i]
    return predictive