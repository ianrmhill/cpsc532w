"""Entry point for the BOED-based analog circuit debug tool."""

import torch as tc
import pyro
from pyro.contrib.oed.eig import marginal_eig
from pyro.optim import Adam
import matplotlib.pyplot as plt

from models.hard_models import dig_boed_guide, dig_guide, make_mdl


class CUT:
    def __init__(self, n1, n2, n3, n4, n5):
        self.n1_f = n1
        self.n2_f = n2
        self.n3_f = n3
        self.n4_f = n4
        self.n5_f = n5

    def run(self, i1, i2, i3):
        n1 = self.n1_f if self.n1_f else i1
        n2 = self.n2_f if self.n2_f else i2
        n3 = self.n3_f if self.n3_f else i3
        n4 = self.n4_f if self.n4_f else int(n1 or n2)
        n5 = self.n5_f if self.n5_f else int(n4 and n3)
        return n5


def eval_test_eigs(mdl, designs, viz_results: bool = False):
    # Now for a BOED phase

    pyro.clear_param_store()
    num_steps, start_lr, end_lr = 1000, 0.1, 0.001
    optimizer = pyro.optim.ExponentialLR({'optimizer': tc.optim.Adam,
                                          'optim_args': {'lr': start_lr},
                                          'gamma': (end_lr / start_lr) ** (1 / num_steps)})

    eig = marginal_eig(
        mdl,
        designs,       # design, or in this case, tensor of possible designs
        ['O'],                  # site label of observations, could be a list
        ['N1-F', 'N2-F', 'N3-F', 'N4-F', 'N5-F'],      # site label of 'targets' (latent variables), could also be list
        num_samples=100,         # number of samples to draw per step in the expectation
        num_steps=num_steps,     # number of gradient steps
        guide=dig_boed_guide,        # guide q(y)
        optim=optimizer,         # optimizer with learning rate decay
        final_num_samples=1000   # at the last step, we draw more samples
        )

    if viz_results:
        plt.figure(figsize=(10,5))
        x_vals = ['000', '001', '010', '011', '100', '101', '110', '111']
        plt.plot(x_vals, eig.detach().numpy(), marker='o', linewidth=2)
        plt.xlabel("Input set")
        plt.ylabel("EIG")
        plt.show()
    return eig.detach()


def amortized_dig_debug():
    test_circuit = CUT(None, None, None, None, None)
    candidate_designs = tc.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                                   [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]], dtype=tc.float)
    fault_priors = tc.tensor([0.8, 0.1, 0.1])
    beliefs = fault_priors.tile((5, 1))
    pyro.clear_param_store()

    for test_pattern in range(3):
        print(f"Beginning round {test_pattern}")
        # Construct current model based on beliefs
        curr_mdl = make_mdl(beliefs)

        # First determine best test pattern to apply
        best = int(tc.argmax(eval_test_eigs(curr_mdl, candidate_designs)).float().detach())

        # Apply the test pattern to the actual circuit
        print(f"Applying test pattern {candidate_designs[best, :]}")
        out = test_circuit.run(*candidate_designs[best, :])
        print(f"Result: {out}")

        # Condition the model based on the outputs
        cond_mdl = pyro.condition(curr_mdl, {'O': tc.tensor(out)})
        svi = pyro.infer.SVI(cond_mdl,
                             dig_guide,
                             Adam({"lr": .05}),
                             loss=pyro.infer.Trace_ELBO(),
                             num_samples=100)
        num_iters = 2000
        for i in range(num_iters):
            elbo = svi.step(candidate_designs[best, :])

        # Update the current model based on the posterior
        for i in range(5):
            beliefs[i, 0] = pyro.param(f"n{i}c").detach().clone()
            beliefs[i, 1] = pyro.param(f"n{i}0").detach().clone()
            beliefs[i, 2] = pyro.param(f"n{i}1").detach().clone()
            print(f"Round {test_pattern} updated beliefs for N{i+1}: {beliefs[i, :]}")


if __name__ == '__main__':
    amortized_dig_debug()
