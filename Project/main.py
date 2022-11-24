"""Entry point for the BOED-based analog circuit debug tool."""

import torch as tc
import pyro

from models.hard_models import res_meas_circuit, res_meas_guide


def main():
    v_ins = tc.tensor([1.9, 2.5, 4.2, 3.3])
    v_outs = tc.tensor([1.25, 1.78, 2.85, 2.23])

    #cond_model = pyro.condition(res_meas_circuit, {'VO': v_outs})
    svi = pyro.infer.SVI(res_meas_circuit,
                   res_meas_guide,
                   pyro.optim.Adam({"lr": .3}),
                   loss=pyro.infer.Trace_ELBO(),
                   num_samples=100)
    pyro.clear_param_store()
    num_iters = 5000
    for i in range(num_iters):
        loss = svi.step(v_ins, v_outs)
        if i % 500 == 0:
            print(f"Curr: R1 = N({pyro.param('R1-mu')}, {pyro.param('R1-sig')}), R2 = N({pyro.param('R2-mu')}, {pyro.param('R2-sig')})")
            print("Neg ELBO:", loss)

    print(f"Posterior: R1 = N({pyro.param('R1-mu')}, {pyro.param('R1-sig')}), R2 = N({pyro.param('R2-mu')}, {pyro.param('R2-sig')})")


if __name__ == '__main__':
    main()
