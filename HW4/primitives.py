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
        i = int(i.item())
    return x[i]


def putindex(x, i, v):
    if type(i) == tc.Tensor:
        i = int(i.item())
    x[i] = v
    return x


def ternary(c, a, b):
    return a if c else b


def transpose(x):
    return tc.transpose(x, 0, 1)


def repeat_matrix(x, rows, cols):
    return x.repeat(int(rows), int(cols))

def screw_your_dirac(x):
    return tc.distributions.Normal(x, 0.001)


# Primitive function dictionary
primitives = {

    # Comparisons
    '<': tc.lt,
    '<=': tc.le,
    '>': tc.gt,
    '>=': tc.ge,
    '=': tc.eq,
    'and': tc.logical_and,
    'or': tc.logical_or,

    # Math
    '+': tc.add,
    '-': tc.sub,
    'sqrt': tc.sqrt,
    '*': tc.mul,
    '/': tc.div,

    # Containers
    'vector': vector,
    'hash-map': hashmap,

    # Matrices
    'mat-mul': tc.matmul,
    'mat-transpose': transpose,
    'mat-add': tc.add,
    'mat-repmat': repeat_matrix,
    'mat-tanh': tc.tanh,

    # Distributions
    'normal': tc.distributions.Normal,
    'beta': tc.distributions.Beta,
    'exponential': tc.distributions.Exponential,
    'uniform-continuous': tc.distributions.Uniform,
    'discrete': tc.distributions.Categorical,
    'bernoulli': tc.distributions.Bernoulli,
    'flip': tc.distributions.Bernoulli,
    'gamma': tc.distributions.Gamma,
    'dirichlet': tc.distributions.Dirichlet,
    'dirac': screw_your_dirac,

    # List Operations
    'get': getindex,
    'put': putindex,
    'first': lambda x: x[0],
    'second': lambda x: x[1],
    'last': lambda x: x[-1],
    'rest': lambda x: x[1:],
    'nth': lambda x, i: x[int(i)-1],
    # Imagine writing a library where you have to do the below just to append to a vector, incredible
    # I know GPUs are picky buggers but come on
    'append': lambda x, v: tc.cat((x, tc.unsqueeze(v, dim=-1)))
}