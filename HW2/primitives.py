import torch as tc

def vector(*x):
    # NOTE: This must support both lists and vectors
    try:
        result = tc.stack(x)
    except:
        result = list(x)
    return result

def hashmap(*x):
    _keys = [key for key in x[0::2]]
    keys = []
    for key in _keys: # Torch tensors cannot be dictionary keys, so convert here
        if type(key) is tc.Tensor: key = float(key)
        keys.append(key)
    values = [value for value in x[1::2]]
    return dict(zip(keys, values))


def getindex(x, i):
    if type(i) == tc.Tensor:
        i = i.item()
    return x[i]


def putindex(x, i, v):
    if type(i) == tc.Tensor:
        i = i.item()
    x[i] = v
    return x


def ternary(c, a, b):
    return a if c else b


# Primitive function dictionary
primitives = {

    # Comparisons
    '<': tc.lt,
    '<=': tc.le,
    'if': ternary,
    # ...

    # Math
    '+': tc.add,
    '-': tc.sub,
    'sqrt': tc.sqrt,
    '*': tc.mul,
    '/': tc.div,

    # Containers
    'vector': vector,
    'hash-map': hashmap,
    # ...

    # Matrices
    'mat-mul': tc.matmul,
    # ...

    # Distributions
    'normal': tc.distributions.Normal,
    # ...

    # List Operations
    'get': getindex,
    'put': putindex,
    'first': lambda x: x[0],
    'last': lambda x: x[-1],
    # Imagine writing a library where you have to do the below just to append to a vector, incredible
    # I know GPUs are picky buggers but come on
    'append': lambda x, v: tc.cat((x, tc.unsqueeze(v, dim=-1)))
}