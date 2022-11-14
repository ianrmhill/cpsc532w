# Standard imports
import torch as tc
from pyrsistent._pmap import PMap
from pyrsistent._plist import PList, _EmptyPList as EmptyPList
from pyrsistent import pmap, plist

def vector(*x):
    # This needs to support both lists and vectors
    try:
        result = tc.stack(x) # NOTE: Important to use stack rather than tc.tensor
    except: # NOTE: This except is horrible, but necessary for list/vector ambiguity
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
    previous_address, current_address, continuation = x[0], x[1], x[2]
    return continuation(previous_address+'-'+current_address)


def get_index(*x):
    if type(x[2]) == str:
        return x[3](x[1][x[2]])
    else:
        return x[3](x[1][int(x[2])])


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
    return x[4](new)


def conjoin(*x):
    if type(x[1]) != tc.Tensor and len(x[1]) == 0:
        return x[3](x[2])
    else:
        if x[1].ndim == 0:
            second = x[1].view(1)
        else:
            second = x[1]
        if x[2].ndim == 0:
            first = x[2].view(1)
        else:
            first = x[2]
        return x[3](tc.cat((first, second)))


def array_empty(*x):
    truth = True if tc.numel(*x[1:-1]) == 0 else False
    return x[-1](truth)


# Primitive function dictionary
# NOTE: Fill this in

# For CPS + SMC there are 2 weirdo arguments in every call. First is the execution path to the 'address', i.e. the
# latent variable we most recently sampled, then the last argument is the continuation, i.e. the rest of the program
# that we call

    # Comparisons
def prim_lt(*x): return x[-1](tc.lt(*x[1:-1]))
def prim_gt(*x): return x[-1](tc.gt(*x[1:-1]))
def prim_ge(*x): return x[-1](tc.ge(*x[1:-1]))
def prim_le(*x): return x[-1](tc.le(*x[1:-1]))
def prim_eq(*x): return x[-1](tc.eq(*x[1:-1]))
def prim_ne(*x): return x[-1](tc.ne(*x[1:-1]))
def prim_empty(*x): return x[-1](True if tc.numel(*x[1:-1]) == 0 else False)

    # Maths
def prim_add(*x): return x[-1](tc.add(*x[1:-1]))
def prim_sub(*x): return x[-1](tc.sub(*x[1:-1]))
def prim_mul(*x): return x[-1](tc.mul(*x[1:-1]))
def prim_div(*x): return x[-1](tc.div(*x[1:-1]))
def prim_sqrt(*x): return x[-1](tc.sqrt(*x[1:-1]))
def prim_log(*x): return x[-1](tc.log(*x[1:-1]))

    # Containers
def prim_vec(*x): return x[-1](vector(*x[1:-1]))
def prim_hash(*x): return x[-1](hashmap(*x[1:-1]))

    # List operations
def prim_first(*x): return x[-1](x[1][0])
def prim_last(*x): return x[-1](x[1][-1])
def prim_rest(*x): return x[-1](x[1][1:])
def prim_append(*x): return x[-1](tc.cat((x[1], x[2].view(1))))
def prim_peek(*x): return x[-1](x[1][0])

    # Matrices
def prim_t(*x): return x[-1](tc.transpose(*x[1:-1], 0, 1))

    # Distributions
def prim_dist_norm(*x): return x[-1](tc.distributions.Normal(*x[1:-1]))
def prim_dist_exp(*x): return x[-1](tc.distributions.Exponential(*x[1:-1]))
def prim_dist_uni(*x): return x[-1](tc.distributions.Uniform(*x[1:-1]))
def prim_dist_beta(*x): return x[-1](tc.distributions.Beta(*x[1:-1]))
def prim_dist_bern(*x): return x[-1](tc.distributions.Bernoulli(*x[1:-1]))
def prim_dist_cat(*x): return x[-1](tc.distributions.Categorical(*x[1:-1]))


primitives = {

    # HOPPL
    'push-address': push_address,

    # Comparisons
    '<': prim_lt,
    '>': prim_gt,
    '>=': prim_ge,
    '<=': prim_le,
    '==': prim_eq,
    '!=': prim_ne,
    'empty?': prim_empty,

    # Maths
    '+': prim_add,
    '-': prim_sub,
    '*': prim_mul,
    '/': prim_div,
    'sqrt': prim_sqrt,
    'log': prim_log,

    # Containers
    'vector': prim_vec,
    'hash-map': prim_hash,

    # List operations
    'get': get_index,
    'put': put_index,
    'first': prim_first,
    'last': prim_last,
    'rest': prim_rest,
    'append': prim_append,
    'conj': conjoin,  # This is NOT conjugate, so dumb
    'peek': prim_peek,

    # Matrices
    'mat-transpose': prim_t,

    # Distributions
    'normal': prim_dist_norm,
    'exponential': prim_dist_exp,
    'uniform-continuous': prim_dist_uni,
    'beta': prim_dist_beta,
    'bernoulli': prim_dist_bern,
    'flip': prim_dist_bern,
    'discrete': prim_dist_cat,

}