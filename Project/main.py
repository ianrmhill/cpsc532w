"""Entry point for the BOED-based analog circuit debug tool."""

import torch as tc
import pyro
from pyro.contrib.oed.eig import marginal_eig
import matplotlib.pyplot as plt

from models.hard_models import res_meas_circuit, res_meas_guide, res_meas_boed, test_guide


def main():
    # v_ins = tc.tensor([1.9, 2.5, 4.2, 3.3])
    # v_outs = tc.tensor([1.25, 1.78, 2.85, 2.23])

    # svi = pyro.infer.SVI(res_meas_circuit,
    #                res_meas_guide,
    #                pyro.optim.Adam({"lr": .3}),
    #                loss=pyro.infer.Trace_ELBO(),
    #                num_samples=100)
    # pyro.clear_param_store()
    # num_iters = 5000
    # for i in range(num_iters):
    #     loss = svi.step(v_ins, v_outs)
    #     if i % 500 == 0:
    #         print(f"Curr: R1 = N({pyro.param('R1-mu')}, {pyro.param('R1-sig')}), R2 = N({pyro.param('R2-mu')}, {pyro.param('R2-sig')})")
    #         print("Neg ELBO:", loss)

    # print(f"Posterior: R1 = N({pyro.param('R1-mu')}, {pyro.param('R1-sig')}), R2 = N({pyro.param('R2-mu')}, {pyro.param('R2-sig')})")

    # Now for a BOED phase
    candidate_designs = tc.arange(0, 4, step=0.2, dtype=tc.float)
    pyro.clear_param_store()
    num_steps, start_lr, end_lr = 1000, 0.1, 0.001
    optimizer = pyro.optim.ExponentialLR({'optimizer': tc.optim.Adam,
                                          'optim_args': {'lr': start_lr},
                                          'gamma': (end_lr / start_lr) ** (1 / num_steps)})

    eig = marginal_eig(
        res_meas_boed,
        candidate_designs,       # design, or in this case, tensor of possible designs
        ['VO'],                  # site label of observations, could be a list
        ['R1', 'R2'],            # site label of 'targets' (latent variables), could also be list
        num_samples=100,         # number of samples to draw per step in the expectation
        num_steps=num_steps,     # number of gradient steps
        guide=test_guide,        # guide q(y)
        optim=optimizer,         # optimizer with learning rate decay
        final_num_samples=1000   # at the last step, we draw more samples
        )

    plt.figure(figsize=(10,5))
    plt.plot(candidate_designs.numpy(), eig.detach().numpy(), marker='o', linewidth=2)
    plt.xlabel("$V_{in}$")
    plt.ylabel("EIG")
    plt.show()


if __name__ == '__main__':
    main()
