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
    def __init__(self):
        pass



def evaluate_program(ast, verbose=False):
    eval_env = {}
    result = interpret(ast.ast_json[0], eval_env)
    return result


def interpret(ast, env):
    # Annoyingly the AST does not already encase expressions that are constant values in lists
    if type(ast) != list:
        ast = [ast]
    # First layer 'q', program can only have one expression, optionally with leading defn statements
    if ast[0] == 'defn':
        # Construct the function
        sub_env = ast[2]
        args = ast[1]
        def thefunc(*args):
            return interpret(ast[3], sub_env)
        env[ast[1]] = thefunc
        # Recurse with the final element 'q' forming the rest of the program
        return interpret(ast[4], env)
    else:
        # Case where expression is a let block
        if ast[0] == 'let':
            env[ast[1][0]] = interpret(ast[1][1], env)[0]
            return interpret(ast[2], env)

        # Case 'c' where expression is just a constant value
        elif type(ast[0]) == int or type(ast[0]) == float:
            return tc.tensor(ast[0]), None, None

        # Case 'c' where expression is a primitive operation
        elif ast[0] in primitives.keys():
            op = primitives[ast[0]]
            for e in range(1, len(ast)):
                ast[e] = interpret(ast[e], env)
            # Pass *args, but need to get only the first element of each return tuple for the args
            return op(*[ast[e][0] for e in range(1, len(ast))]), None, None

        # Cases 'v' and 'f' where expression is a variable or user defined function
        elif ast[0] in env.keys():
            # Case 'v'
            if len(ast) == 1:
                return env[ast[0]], None, None
            # Case 'f'
            else:
                # First add all argument values to the local scope
                for i in range(ast[1]):
                    env[ast[0]][1][i] = ast[1][i]
                # Now run the procedure using the set up local scope
                return env[ast[0]][0](ast[1], env[ast[0][1]])

        # Default case should error out
        else:
            raise Exception('Oh no not yet implemented')
