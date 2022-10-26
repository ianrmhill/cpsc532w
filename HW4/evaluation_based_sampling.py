# Standard imports
import torch as tc

# Project imports
from primitives import primitives # NOTE: Import and use this!


class abstract_syntax_tree:
    def __init__(self, ast_json):
        if type(ast_json) == int:
            ast_json = [[ast_json]]
        self.ast_json = ast_json
        # NOTE: You need to write this!
        # No I don't???


class YummyEnv:
    def __init__(self, init_vars: dict = None, parent_env=None):
        if init_vars:
            self.vars = init_vars.copy()
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


def evaluate_program(ast, sigma, verbose=False):
    to_eval = ast.ast_json
    eval_env = YummyEnv(primitives)
    # Layer 'q', handle any 'defn' statements
    for i in range(len(to_eval) - 1):
        # Construct the function with its child environment
        def user_func(*args, **extras):
            sub_env = YummyEnv({to_eval[i][2][j]: args[j] for j in range(len(to_eval[i][2]))}, extras['env'])
            return interpret(to_eval[i][3], extras['sigma'], sub_env)

        # Add the new function to the environment
        eval_env.add(to_eval[0][1], user_func)

    result, sig, _ = interpret(to_eval[-1], sigma, eval_env)
    return result, sig


def interpret(ast, sigma, env, mode: str = 's'):
    # Annoyingly the AST does not already encase expressions that are constant values in lists
    if type(ast) != list:
        ast = [ast]

    # Case where expression is a let block
    if ast[0] == 'let':
        val, sig, e = interpret(ast[1][1], sigma, env, mode)
        env.add(ast[1][0], val)
        return interpret(ast[2], sig, e)

    # Case where we are obtaining a sample from a distribution
    elif ast[0] == 'sample' or ast[0] == 'sample*':
        if mode == 's':
            dist, sig, e = interpret(ast[1], sigma, env, mode)
            return dist.sample().float(), sig, e
        else:
            dist, sig, e = interpret(ast[1], sigma, env, mode)
            try:
                sig['logw'] += dist.log_prob(env.retrieve('*sample_val*'))
            except ValueError:
                # This case used for VI with annoying variational support issues. If the sample value is outside the PDF
                # support the probability is 0, and so we just return a super small value to approximate 0
                # Used for Q5
                sig['logw'] += -300
            return env.retrieve('*sample_val*'), sig, e

    # Case where we are observing a random variable
    elif ast[0] == 'observe' or ast[0] == 'observe*':
        dist, sig1, e1 = interpret(ast[1], sigma, env, mode)
        obs, sig2, e2 = interpret(ast[2], sig1, e1, mode)
        sig2['logw'] = sig2['logw'] + dist.log_prob(obs)
        return obs, sig2, e2

    # Case 'if' where expression is the ternary
    elif ast[0] == 'if':
        truth, sig, e = interpret(ast[1], sigma, env, mode)
        if truth:
            return interpret(ast[2], sig, e, mode)
        else:
            return interpret(ast[3], sig, e, mode)

    # Case 'c' where expression is just a constant value
    elif type(ast[0]) in [int, float, bool]:
        if type(ast[0]) != bool:
            return tc.tensor(ast[0], dtype=tc.float), sigma, env
        else:
            return tc.tensor(int(ast[0]), dtype=tc.float), sigma, env

    # Cases for variables and functions where expression is a variable, primitive, or user defined function
    elif env.check_for(ast[0]):
        # Case 'v'
        if len(ast) == 1:
            return env.retrieve(ast[0]), sigma, env
        # Case 'f' and 'c' where 'c' is a primitive function
        else:
            # Evaluate and construct the function argument list
            args, sig, e = [], sigma, env
            for arg in ast[1:]:
                val, sig, e = interpret(arg, sig, e, mode)
                args.append(val)
            # If a primitive operation, no observe statements are possible so sigma won't be modified
            if ast[0] in primitives:
                op = env.retrieve(ast[0])
                return op(*args), sig, e
            # User defined functions may result in side effects via 'observe' statements
            else:
                op = env.retrieve(ast[0])
                return op(*args, sigma=sig, env=e)

    # Default case should error out
    else:
        raise Exception('Oh no not yet implemented')
