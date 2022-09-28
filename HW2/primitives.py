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

# Primative function dictionary
# NOTE: You should complete this
primitives = {

    # Comparisons
    '<': tc.lt,
    '<=': tc.le,
    # ...

    # Math
    '+': tc.add,
    '-': tc.sub,
    # ...

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

}