# Standard imports
import torch as tc
from pyrsistent import pmap
from time import time

# Project imports
from primitives import primitives

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
    def __init__(self, params: list, body: list, sig: dict, env: Env, mode: str):
        self.params, self.body, self.sig, self.env, self.mode = params, body, sig, env, mode

    def __call__(self, *args):
        return hoppl_eval(self.body, self.sig, Env(self.params, args, self.env), self.mode)


def standard_env():
    'An environment with some standard procedures'
    env = Env()
    env.update(primitives)
    return env


def hoppl_eval(e, sig:dict, env:Env, mode='prior', verbose=False):
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
            # Even dumber catch case because address names aren't encased in a second pair of quotes for some reason
            elif e[0:4] == 'addr':
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
        return Procedure(prms, expr, sig, env, mode)
    # Now for the if statement which is handled in a special way to pass side effects correctly
    elif e[0] == 'if':
        cond = hoppl_eval(e[1], sig, env, mode)
        # Note the lazy evaluation
        if cond:
            return hoppl_eval(e[2], sig, env, mode)
        else:
            return hoppl_eval(e[3], sig, env, mode)
    # Next are the probabilistic programming constructs, first sample
    elif e[0] == 'sample':
        addr = e[1]
        dist = hoppl_eval(e[2], sig, env, mode)
        continuation = hoppl_eval(e[3], sig, env, mode)
        # Instead of directly running the continuation sample and observe 'break' out of the execution and let the
        # outer evaluate function handle the continuation. This enables trampolining to avoid stack explosion and SMC
        if mode == 'prior':
            return continuation, [dist.sample().float()], sig
        elif mode == 'smc':
            return 'sample', addr, dist, dist.sample().float(), continuation
        else:
            raise Exception('Mode not yet implemented!')
    # Now observe!
    elif e[0] == 'observe':
        addr = e[1]
        dist = hoppl_eval(e[2], sig, env, mode)
        obs = hoppl_eval(e[3], sig, env, mode)
        continuation = hoppl_eval(e[4], sig, env, mode)
        # For now ignore the observed value as we're just trying to get the prior
        if mode == 'prior':
            return continuation, [dist.sample().float(), obs], sig
        elif mode == 'smc':
            return 'observe', addr, dist, obs, continuation
        else:
            raise Exception('Mode not yet implemented!')
    # Everything else is just summoning some expression by name from the environment
    else:
        if type(e[0]) == list:
            # Function 'fn' to compile
            func = hoppl_eval(e[0], sig, env, mode)
        else:
            # Primitive
            func = env.find(e[0])[e[0]]
        # Now evaluate all the arguments, except the passed address which is just immediately added to the args list
        args = []
        for arg in e[1:]:
            val = hoppl_eval(arg, sig, env, mode)
            args.append(val)
        # Note that env.find() returns the innermost environment where the requested variable is found
        # We then get the value of the requested variable, and call it with all the arguments
        tada = func(*args)
        return tada


def evaluate(ast:dict, sig=None, run_name='start', verbose=False):
    '''
    Evaluate a HOPPL program as desugared by daphne
    Args:
        ast: abstract syntax tree
    Returns: The return value of the program
    '''
    if not sig:
        sig = {}
    env = standard_env()
    output = lambda x: x # Identity function, so that output value is identical to output
    exp = hoppl_eval(ast, sig, env, verbose)(run_name, output) # NOTE: Must run as function with a continuation
    while type(exp) is tuple: # If there are continuations the exp will be a tuple and a re-evaluation needs to occur
        continuation, args, sig = exp
        exp = continuation(*args)
    return exp, sig