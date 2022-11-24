"""Hard-coded circuit and experiment models for pre-spice integration BOED."""

import torch as tc
from torch.distributions import constraints
import pyro
import pyro.distributions as dist

R1 = tc.tensor(4, dtype=tc.float)
R2 = tc.tensor(8, dtype=tc.float)


def defected_res_net(v_in):
    # Define the latent variables that determine possible faults and component values, but NOT circuit topology
    r_1 = pyro.sample("R1", dist.Normal(R1, 100.)) # 5% resistor value tolerances
    r_2 = pyro.sample("R2", dist.Normal(R2, 200.))
    r_1_shorted = pyro.sample("R1-S", dist.Bernoulli(0.2))
    r_1_open = pyro.sample("R1-O", dist.Bernoulli(0.2))
    #r_2_shorted = pyro.sample("R2-S", dist.Bernoulli(0.1))
    #r_2_open = pyro.sample("R2-O", dist.Bernoulli(0.1))

    # Define the link functions that determine v_o based on the components and any occurring faults
    v_o = v_in * (r_2 / (r_1 + r_2))
    if r_1_open:
        v_o = v_in * 0.
    elif r_1_shorted:
        v_o = v_in
    return pyro.sample("VO", dist.Normal(v_o, 0.001).to_event(1))


def res_net_guide(v_in):
    r1_mu = pyro.param('R1-mu', tc.tensor(4000.))
    r1_sig = pyro.param('R1-sig', tc.tensor(1000.))
    r2_mu = pyro.param('R2-mu', tc.tensor(8000.))
    r2_sig = pyro.param('R2-sig', tc.tensor(1000.))
    pyro.sample('R1', dist.Normal(r1_mu, r1_sig))
    pyro.sample('R2', dist.Normal(r2_mu, r2_sig))
    r1_s_prior = pyro.param('R1-S-prob', tc.tensor(0.5), constraint=constraints.interval(0., 1.))
    r1_o_prior = pyro.param('R1-O-prob', tc.tensor(0.5), constraint=constraints.interval(0., 1.))
    pyro.sample('R1-S', dist.Bernoulli(r1_s_prior))
    pyro.sample('R1-O', dist.Bernoulli(r1_o_prior))


def res_meas_circuit(v_in, v_o_obs):
    # Define the latent variables that determine possible faults and component values, but NOT circuit topology
    r_1 = pyro.sample('R1', dist.Normal(R1, 0.1)) # 5% resistor value tolerances
    r_2 = pyro.sample('R2', dist.Normal(R2, 0.2))

    # Define the link functions that determine v_o based on the components and any occurring faults
    v_o = v_in * (r_2 / (r_1 + r_2))
    return pyro.sample('VO', dist.Normal(v_o, 0.001).to_event(1), obs=v_o_obs)


def res_meas_guide(v_in, v_o_obs):
    r1_mu = pyro.param('R1-mu', tc.tensor(4.))
    r1_sig = pyro.param('R1-sig', tc.tensor(1.), constraint=constraints.positive)
    r2_mu = pyro.param('R2-mu', tc.tensor(8.))
    r2_sig = pyro.param('R2-sig', tc.tensor(2.), constraint=constraints.positive)
    pyro.sample('R1', dist.Normal(r1_mu, r1_sig))
    pyro.sample('R2', dist.Normal(r2_mu, r2_sig))


def test_guide(design, observation_labels, target_labels):
    pyro.param('VO-mu', tc.tensor(2.2))
