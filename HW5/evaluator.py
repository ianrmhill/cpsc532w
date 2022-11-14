# Standard imports
import torch as tc
from time import time

# Project imports
from primitives import primitives
from utils import log_sample
from pyrsistent import PMap

# Parameters
run_name = 'start'

class Env(dict):
    'An environment: a dict of {var: val} pairs, with an outer environment'
    def __init__(self, parms=(), args=(), outer=None):
        self.update(zip(parms, args))
        self.outer = outer
        super().__init__()

    def find(self, var):
        'Find the innermost Env where var appears'
        if var in self:
            result = self
        else:
            if self.outer is None:
                print('Not found in any environment:', var)
                raise ValueError('Outer limit of environment reached')
            else:
                result = self.outer.find(var)
        return result


class Procedure(object):
    'A user-defined HOPPL procedure'
    def __init__(self, params:list, body:list, sig:dict, env:Env):
        self.params, self.body, self.sig, self.env = params, body, sig, env

    def __call__(self, *args):
        return hoppl_eval(self.body, self.sig, Env(self.params, args, self.env))


def standard_env():
    'An environment with some standard procedures'
    env = Env()
    env.update(primitives)
    return env


# Renamed to avoid shadowing the built-in 'eval'
def hoppl_eval(e, sig:dict, env:Env, verbose=False):
    '''
    The eval routine
    @params
        e: expression
        sig: side-effects
        env: environment
    '''
    # It's recursion time baby, just turtles on turtles
    # First case is constants and variables. I don't like checking ast type on principle, but oh well it works
    if type(e) != list:
        # Case variable
        if type(e) == str:
            # Stupid catch case as hash maps might use string literal keys
            if e[0] == "\"":
                return e
            return env.find(e)[e]
        # Case constant
        else:
            # Need the silly bool casting again, everything's gotta be tensors or torch acts up
            if type(e) == bool:
                e = int(e)
            return tc.tensor(e, dtype=tc.float)
    # Next up is function/procedure definitions! This is where the fun begins
    elif e[0] == 'fn':
        prms, expr = e[1], e[2]
        #expr, n_sig, n_env = hoppl_eval(e[2], sig, env)
        return Procedure(prms, expr, sig, env)
    # Now for the if statement which is handled in a special way to pass side effects correctly
    elif e[0] == 'if':
        cond = hoppl_eval(e[1], sig, env)
        # Note the lazy evaluation
        if cond:
            return hoppl_eval(e[2], sig, env)
        else:
            return hoppl_eval(e[3], sig, env)
    # Next are the probabilistic programming constructs, first sample
    elif e[0] == 'sample':
        addr = e[1]
        dist = hoppl_eval(e[2], sig, env)
        return dist.sample().float()
    # Now observe!
    elif e[0] == 'observe':
        addr = e[1]
        dist = hoppl_eval(e[2], sig, env)
        obs = hoppl_eval(e[3], sig, env)
        # For now ignore the observed value as we're just trying to get the prior
        return dist.sample().float()
    # Everything else is just summoning some expression by name from the environment
    else:
        if type(e[0]) == list:
            # Function 'fn' to compile
            func = hoppl_eval(e[0], sig, env)
        else:
            # Primitive
            func = env.find(e[0])[e[0]]
        # Now evaluate all the arguments, except the passed address which is just immediately added to the args list
        args = [e[1]]
        for arg in e[2:]:
            val = hoppl_eval(arg, sig, env)
            args.append(val)
        # Note that env.find() returns the innermost environment where the requested variable is found
        # We then get the value of the requested variable, and call it with all the arguments
        tada = func(*args)
        return tada


def evaluate(ast:dict, verbose=False):
    '''
    Evaluate a HOPPL program as desugared by daphne
    Args:
        ast: abstract syntax tree
    Returns: The return value of the program
    '''
    sig = {}; env = standard_env()
    program = hoppl_eval(ast, sig, env, verbose)
    val = program(run_name) # NOTE: Must run as function with *any* argument
    if type(val) == PMap:
        val = dict(val)
    return val


def get_samples(ast:dict, num_samples:int, tmax=None, wandb_name=None, verbose=False):
    '''
    Generate a set of samples from the prior of a FOPPL program
    '''
    samples = []
    if (tmax is not None): max_time = time()+tmax
    for i in range(num_samples):
        sample = evaluate(ast, verbose)
        if wandb_name is not None: log_sample(sample, i, wandb_name=wandb_name)
        samples.append(sample)
        if (tmax is not None) and time() > max_time: break
    return samples
