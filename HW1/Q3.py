import torch
import torch.distributions as dist
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import wandb
import numpy as np
import random

# UBC CS532W H1 Q3


# first define the probability distributions as defined in the exercise:

# Use 0 as false, 1 as true
# You should use type hinting throughout and familiarize yourself with PyTorch tensors and distributions
def p_C(c:torch.Tensor) -> torch.Tensor:
    probs = torch.tensor([0.5,0.5])
    d = dist.Categorical(probs)
    return torch.exp(d.log_prob(c))


def p_S_given_C(s: torch.Tensor,c: torch.Tensor) -> torch.Tensor:
    probs = torch.tensor([[0.5, 0.9], [0.5, 0.1]])
    d = dist.Categorical(probs.t())
    lp = d.log_prob(s)[c.detach()] 
    return torch.exp(lp)


def p_R_given_C(r: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    probs = torch.tensor([[0.8, 0.2], [0.2, 0.8]])
    d = dist.Categorical(probs.t())
    lp = d.log_prob(r)[c.detach()]  
    return torch.exp(lp)


def p_W_given_S_R(w: torch.Tensor, s: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    probs = torch.tensor([
        [[1.0, 0.1], [0.1, 0.01]],  # w = False
        [[0.0, 0.9], [0.9, 0.99]],  # w = True
    ])
    return probs[w.detach(), s.detach(), r.detach()]


# we will be using hydra throughout the course to control configurations
# all arguments are command-line overrideable and you should definitely
# look at the contents of ./conf/config.yaml 
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def my_app(cfg: DictConfig) -> None:
    # in general you should use logging instead of simply printing
    # to the terminal
    log = logging.getLogger(__name__)

    # this is how you get configuration settings from the yaml via hydra
    wandb_entity = cfg['wandb_entity']
    wandb_project = cfg['wandb_project']
    wandb_logging = cfg['wandb_logging']
    seed = cfg['seed']

    # you should always control the seed in pseudo-random experiments
    # but be aware that GPU computation is not deterministic
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # you should let wandb log your code so that it is automatically turned-in
    # leaving the wandb_project set as it is in the yaml will let you "collaborate"
    # with others in the class working on the same project
    if wandb_logging:
        # set up wandb
        wandb.init(project=wandb_project, entity=wandb_entity)
        wandb.run.log_code(".")
        wandb.config.update(OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        ))


    ##1. enumeration and conditioning:

    ## compute joint:
    p = torch.zeros((2,2,2,2)) #c,s,r,w
    for c in range(2):
        for s in range(2):
            for r in range(2):
                for w in range(2):
                    p[c, s, r, w] = p_C(torch.tensor(c))*p_S_given_C(torch.tensor(s), torch.tensor(c))*p_R_given_C(torch.tensor(r), torch.tensor(c))*p_W_given_S_R(torch.tensor(w), torch.tensor(s), torch.tensor(r))
    
    # NOTE: Fill this in...
    # Observing that w is 1 (i.e. it's true that the grass is wet, we simply sum the joint probabilities across all dims
    # except for w, then read out the sum for w=1. Dim index of w is 3.
    d = torch.index_select(p, 3, torch.tensor(1))
    d_sum = torch.sum(d, dim=(1, 2))
    d_nottorch = np.array(d_sum)
    odds = float(d_nottorch[1] / (np.sum(d_nottorch)))

    log.info('#1 (enum and cond) : There is a {:.2f}% chance it is cloudy given the grass is wet'.format(odds*100))

    if wandb_logging:
        wandb.log({'prob_cloudy_given_wet_grass': {
                  'enum and cond': odds*100}})

    # things to think about here: when can you enumerate?  and what
    # is the computational complexity of enumeration?

    ##2. ancestral sampling and rejection:
    num_samples = 10000
    samples = torch.zeros(num_samples)
    rejections = 0
    i = 0

    # NOTE: Fill this in
    while i < num_samples:
        # Generate a sample from p(x)
        p_flat = torch.flatten(p)
        x = dist.Categorical(p_flat).sample()
        w = x % 2
        r = ((x - w) / 2) % 2
        s = ((x - 2*r - w) / 4) % 2
        c = ((x - 4*s - 2*r - w) / 8) % 2
        x = torch.tensor([c, s, r, w])
        # Sample u from the uniform distribution on [0, q(x)]
        # u = dist.Uniform(0, 1).sample()
        # Calculate p(x) using the enumerated joint and accept or reject, but we're gonna shortcut since we know
        # we will only and always accept when w = 1
        if x[3] == 1:
            samples[i] = x[0]
            i += 1
        else:
            rejections += 1

    log.info('#2 (ancestral + reject) : The chance of it being cloudy given the grass is wet is {:.2f}%'.format(samples.mean()*100))
    log.info('#2 (ancestral + reject) : {:.2f}% of the total samples were rejected'.format(
        100*rejections/float(num_samples+rejections))) # FIXED

    # things to think about here: when will rejection sampling be efficient or 
    # even possible?  can you rejection sample if the conditioning event is
    # measure zero?  what if the conditioning event is extremely unlikely?

    if wandb_logging:
        wandb.log({'prob_cloudy_given_wet_grass': {
                  'ancestral + reject': samples.mean()*100}})

    #3: Gibbs
    # we can use the joint above to condition on the variables, to create the needed
    # conditional distributions:

    #we can calculate p(R|C,S,W) and p(S|C,R,W) from the joint, dividing by the right marginal distribution
    #indexing is [c,s,r,w]
    p_R_given_C_S_W = p/p.sum(axis=2, keepdims=True)
    p_S_given_C_R_W = p/p.sum(axis=1, keepdims=True)


    # but for C given R,S,W, there is a 0 in the joint (0/0), arising from p(W|S,R)
    # but since p(W|S,R) does not depend on C, we can factor it out:
    #p(C | R, S) = p(R,S,C)/(int_C (p(R,S,C)))

    #first create p(R,S,C):
    p_C_S_R = torch.zeros((2,2,2)) #c,s,r
    for c in range(2):
        for s in range(2):
            for r in range(2):
                p_C_S_R[c, s, r] = p_C(torch.tensor(c).int())*p_S_given_C(torch.tensor(s).int(), torch.tensor(c).int())*p_R_given_C(torch.tensor(r).int(), torch.tensor(c).int())

    #then create the conditional distribution:
    p_C_given_S_R = p_C_S_R[:,:,:]/p_C_S_R[:,:,:].sum(axis=(0),keepdims=True)

    ##Gibbs sampling
    num_samples = 11000
    samples = torch.zeros(num_samples)
    state = torch.zeros(4, dtype=torch.int8)
    #c,s,r,w, set w = True

    # NOTE: Fill this in
    i=0
    good_samples = []
    while i < num_samples:
        p_C_curr = p_C_given_S_R[:, state[1], state[2]]
        state[0] = dist.Categorical(p_C_curr).sample()
        p_S_curr = p_S_given_C_R_W[state[0], :, state[2], state[3]]
        state[1] = dist.Categorical(p_S_curr).sample()
        p_R_curr = p_R_given_C_S_W[state[0], state[1], :, state[3]]
        state[2] = dist.Categorical(p_R_curr).sample()
        p_W_curr = p[state[0], state[1], state[2], :]
        state[3] = dist.Categorical(p_W_curr).sample()

        joint_prob = p[state[0], state[1], state[2], state[3]]
        samples[i] = state[3]
        if state[3] == 1:
            good_samples.append(state[0].item())
        i += 1

        if wandb_logging:
            wandb.log({'gibbs':{'iteration':i, 'c':state[0], 's':state[1], 'r':state[2], 'w':state[3], 'p(c,s,r,w)':joint_prob}})

    # NOTE: Fill this in
    pcgwg = np.mean(good_samples) * 100
    if wandb_logging:
        wandb.log({'prob_cloudy_given_wet_grass':{'gibbs':pcgwg}})

    log.info('#3 (Gibbs) : The chance of it being cloudy given the grass is wet is {:.2f}%'.format(pcgwg))

    # things to think about here: can you always derive the exact conditionals required by
    # the Gibbs sampling algorithm?  what could you do if you can't? (HW 3)  what happens 
    # if a group of variables is very tightly coupled in the posterior?  will Gibbs sampling
    # be efficient in that case?  what would you do to solve such a problem?

if __name__ == "__main__":
    my_app()
