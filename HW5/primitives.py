# Standard imports
import torch as tc
from pyrsistent import pmap, plist, l

def vector(*x):
    # This needs to support both lists and vectors
    try:
        result = tc.stack(x) # NOTE: Important to use stack rather than tc.tensor
    except:
        result = plist(x)
    return result


def hashmap(*x):
    # This is a dictionary
    keys, values = x[0::2], x[1::2]
    checked_keys = []
    for key in keys: # Torch tensors cannot be dictionary keys, so convert here
        if type(key) is tc.Tensor: key = float(key)
        checked_keys.append(key)
    dictionary = dict(zip(checked_keys, values))
    hashmap = pmap(dictionary)
    return hashmap


def push_address(*x):
    # Concatenate two addresses to produce a new, unique address
    previous_address, current_address = x[0], x[1]
    new_address = previous_address+'-'+current_address
    return new_address


def get_index(*x):
    if type(x[2]) == str:
        return x[1][x[2]]
    else:
        return x[1][int(x[2])]


def put_index(*x):
    new = x[1]
    if type(new) == tc.Tensor:
        new[int(x[2])] = x[3]
    else:
        # String key
        if type(x[2]) == str:
            new = new.set(x[2], x[3])
        # Numeric key
        else:
            new = new.set(x[2].item(), x[3])
    return new


def conjoin(*x):
    if type(x[1]) != tc.Tensor and len(x[1]) == 0:
        return x[2]
    else:
        if x[1].ndim == 0:
            second = x[1].view(1)
        else:
            second = x[1]
        if x[2].ndim == 0:
            first = x[2].view(1)
        else:
            first = x[2]
        return tc.cat((first, second))


# Primitive function dictionary
# NOTE: Fill this in
primitives = {

    # HOPPL
    'push-address' : push_address,

    # Comparisons
    '<': lambda *x: tc.lt(*x[1:]),
    '>': lambda *x: tc.gt(*x[1:]),
    '>=': lambda *x: tc.ge(*x[1:]),
    '<=': lambda *x: tc.le(*x[1:]),
    '==': lambda *x: tc.eq(*x[1:]),
    '!=': lambda *x: tc.ne(*x[1:]),
    'empty?': lambda *x: True if tc.numel(*x[1:]) == 0 else False,

    # Math
    '+': lambda *x: tc.add(*x[1:]),
    '-': lambda *x: tc.sub(*x[1:]),
    '*': lambda *x: tc.mul(*x[1:]),
    '/': lambda *x: tc.div(*x[1:]),
    'sqrt': lambda *x: tc.sqrt(*x[1:]),
    'log': lambda *x: tc.log(*x[1:]),

    # Containers
    'vector': lambda *x: vector(*x[1:]),
    'hash-map': lambda *x: hashmap(*x[1:]),

    # List operations
    'get': get_index,
    'put': put_index,
    'first': lambda *x: x[1][0],
    'last': lambda *x: x[1][-1],
    'rest': lambda *x: x[1][1:],
    'append': lambda *x: tc.cat((x[1], x[2].view(1))),
    'conj': conjoin, # This is NOT conjugate, so dumb
    'peek': lambda *x: x[1][0],

    # Matrices
    'mat-transpose': lambda *x: tc.transpose(*x[1:], 0, 1),

    # Distributions
    'normal': lambda *x: tc.distributions.Normal(*x[1:]),
    'exponential': lambda *x: tc.distributions.Exponential(*x[1:]),
    'uniform-continuous': lambda *x: tc.distributions.Uniform(*x[1:]),
    'beta': lambda *x: tc.distributions.Beta(*x[1:]),
    'bernoulli': lambda *x: tc.distributions.Bernoulli(*x[1:]),
    'flip': lambda *x: tc.distributions.Bernoulli(*x[1:]),
    'discrete': lambda *x: tc.distributions.Categorical(*x[1:]),

}