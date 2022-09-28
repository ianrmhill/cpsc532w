# Standard imports
import torch as tc

# Project imports
from primitives import primitives # NOTE: Import and use this!

class abstract_syntax_tree:
    def __init__(self, ast_json):
        self.ast_json = ast_json
        # NOTE: You need to write this!
        # No I don't???


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
        env[ast[1]] = ast[3](ast[2])
        # Recurse with the final element 'q' forming the rest of the program
        return interpret(ast[4:], env)
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
            return env[ast[0]], None, None

        # Default case should error out
        else:
            raise Exception('Oh no not yet implemented')
