import torch as tc

import distributions as dist
import utils
from graph_based_sampling import eval_graph_prob, eval_graph_given_samples


class Guide:
    def __init__(self, graph, prog_name):
        # Hard coded guides seems lazy, but actually makes sense since the probabilistic programmer wants control over
        # the form of the guide and such. I'm the one that's lazy.
        self.multi_out = False
        match prog_name:
            case 'Program 1':
                self.dists = {'sample2': dist.Normal(tc.tensor(1.), tc.sqrt(tc.tensor(5.)))}
            case 'Program 2':
                self.dists = {'sample1': dist.Normal(tc.tensor(0.), tc.tensor(10.)),
                              'sample2': dist.Normal(tc.tensor(0.), tc.tensor(10.))}
            case 'Program 3':
                self.dists = {f"sample{i}": dist.Normal(tc.tensor(0.), tc.tensor(10.)) for i in range(0, 5, 2)}
                self.dists = self.dists | {f"sample{i}": dist.Gamma(tc.tensor(1.), tc.tensor(1.)) for i in range(1, 6, 2)}
                self.dists['sample6'] = dist.Dirichlet(tc.tensor([1., 1., 1.]))
                self.dists = self.dists | {f"sample{i}": dist.Categorical(tc.tensor([0.3, 0.3, 0.4])) for i in range(7, 20, 2)}
                # Dirichlet outputs 3 values so gotta handle things differently
                self.multi_out = True
            case 'Program 4':
                self.dists = {f"sample{i}": dist.Normal(tc.tensor(0.), tc.tensor(1.)) for i in range(142)}
            case 'Program 5':
                self.dists = {'sample1': dist.Normal(tc.tensor(0.), tc.tensor(5.)),
                              'sample2': dist.Gamma(tc.tensor(1.4), tc.tensor(2.))}
            case _:
                raise Exception()
        self.node_order = [node for node in self.dists]
        self.num_nodes = len(self.node_order)

    def sample(self, num_samples):
        if self.multi_out:
            samples = tc.zeros((self.num_nodes + 2, num_samples))
        else:
            samples = tc.zeros((self.num_nodes, num_samples))
        offset = 0
        for i, node in enumerate(self.node_order):
            ia = i + offset
            if type(self.dists[node]) == dist.Dirichlet:
                samples[ia:ia+3] = self.dists[node].sample(sample_shape=(num_samples,)).T
                offset += 2
            else:
                samples[ia] = self.dists[node].sample(sample_shape=(num_samples,))
        return samples

    def log_prob(self, samples, num_samples):
        logps = tc.zeros((num_samples,))
        offset = 0
        for i, node in enumerate(self.dists):
            ia = i + offset
            if type(self.dists[node]) == dist.Dirichlet:
                logps += self.dists[node].log_prob(samples[ia:ia+3].T)
                offset += 2
            else:
                try:
                    logps += self.dists[node].log_prob(samples[ia])
                except ValueError:
                    # Add awful log prob if we go out of the distribution support
                    logps += -100
        return logps

    def optim_prms(self):
        prms = []
        for i, node in enumerate(self.dists):
            prms.extend(self.dists[node].optim_params())
        return prms


def variational_inference(p_graph, guide, wandb_name, wandb_run, num_epochs=10, samples_per_epoch=10):
    # Initialize things
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
            tofix = []
            if wandb_name == 'Program 3': tofix.append('sample6')
            graph_logp[s] = eval_graph_prob(p_graph.sorted, p_graph.links, guide.node_order, z[:, s], hardfix=tofix)

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
            if wandb_name in ['Program 1', 'Program 2', 'Program 5']:
                utils.log_params(guide.dists, i, wandb_name)
        i += 1

    # Return the trained guide, I've made it mutable
    return guide


def sample_posterior(graph, posterior_guide, num_samples, wandb_name):
    samples = posterior_guide.sample(num_samples)
    hardfix = []
    if wandb_name == 'Program 3': hardfix.append('sample6')
    outputs = []
    for i in range(num_samples):
        outputs.append(eval_graph_given_samples(graph, posterior_guide.node_order, samples[:, i], hardfix))
    if wandb_name == 'Program 4':
        for i in range(len(outputs)):
            outputs[i][2] = tc.reshape(outputs[i][2], (100, 1))
            outputs[i] = tc.cat((outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3], outputs[i][4]), dim=0)
    return tc.stack(outputs)