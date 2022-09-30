# Standard imports
import torch as tc

# Project imports
from primitives import primitives # NOTE: Import and use this!

class abstract_syntax_tree:
    def __init__(self, ast_json):
        self.ast_json = ast_json
        # NOTE: You need to write this!
        # No I don't???


class YummyEnv:
    def __init__(self, init_vars: dict = None, parent_env=None):
        if init_vars:
            self.vars = init_vars
        else:
            self.vars = {}
        self.parent = parent_env

    def check_for(self, var_name: str):
        if var_name in self.vars.keys():
            return True
        elif self.parent:
            return self.parent.check_for(var_name)
        else:
            return False

    def retrieve(self, var_name: str):
        if var_name in self.vars.keys():
            return self.vars[var_name]
        elif self.parent:
            return self.parent.retrieve(var_name)
        else:
            raise Exception(f"Variable {var_name} not defined.")

    def add(self, var_name: str, var_val):
        self.vars[var_name] = var_val


def evaluate_program(ast, verbose=False):
    to_eval = ast.ast_json
    eval_env = YummyEnv(primitives)
    # Layer 'q', handle any 'defn' statements
    for i in range(len(to_eval) - 1):
        # Construct the function with its child environment
        def user_func(*args):
            sub_env = YummyEnv({to_eval[i][2][j]: args[j] for j in range(len(to_eval[i][2]))}, eval_env)
            return interpret(to_eval[i][3], sub_env)[0]

        # Add the new function to the environment
        eval_env.add(to_eval[0][1], user_func)

    result = interpret(to_eval[-1], eval_env)
    return result


def interpret(ast, env):
    # Annoyingly the AST does not already encase expressions that are constant values in lists
    if type(ast) != list:
        ast = [ast]

    # Case where expression is a let block
    if ast[0] == 'let':
        env.add(ast[1][0], interpret(ast[1][1], env)[0])
        return interpret(ast[2], env)

    # Case where we are obtaining a sample from a distribution
    elif ast[0] == 'sample':
        return interpret(ast[1], env)[0].sample(), None, None

    # Case where we are observing a random variable
    elif ast[0] == 'observe':
        # For HW2 we ignore the conditioning expression, just return a distribution sample
        return interpret(ast[1], env)[0].sample(), None, None

    # Case 'c' where expression is just a constant value
    elif type(ast[0]) in [int, float, bool]:
        if type(ast[0]) != bool:
            return tc.tensor(ast[0], dtype=tc.float), None, None
        else:
            return tc.tensor(int(ast[0]), dtype=tc.float), None, None

    # Cases 'v' and 'f' where expression is a variable, primitive, or user defined function
    elif env.check_for(ast[0]):
        # Case 'v'
        if len(ast) == 1:
            return env.retrieve(ast[0]), None, None
        # Case 'f' and 'c' where 'c' is a primitive function
        else:
            # First interpret and add all argument values to the arg list
            f_args = [interpret(arg, env)[0] for arg in ast[1:]]
            # Now run the procedure using the interpreted args
            op = env.retrieve(ast[0])
            return op(*f_args), None, None

    # Default case should error out
    else:
        raise Exception('Oh no not yet implemented')
