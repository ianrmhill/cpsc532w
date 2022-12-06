
import torch as tc

__all__ = ['get_comp_nodes', 'get_eqn']


def get_comp_nodes(comp_name, comp_type):
    match comp_type:
        case 'v_in': return [comp_name]
        case 'v_out': return [comp_name]
        case 'res': return [f"{comp_name}-1", f"{comp_name}-2"]
        case _: raise Exception(f"Unknown component type {comp_type}!")


def get_eqn(node_name, comp_name, comp_type, nodes, conns, prms):
    match comp_type:
        # If input voltage, will just be equal to 1 times itself, no connections influence the value
        case 'v_in':
            eqn = tc.zeros(len(nodes), dtype=tc.float)
            for i in range(len(nodes)):
                if nodes[i] == node_name:
                    eqn[i] = 1

        # Output voltage nodes are the generic KCL case, no special behaviour needed
        case 'v_out':
            eqn = tc.zeros(len(nodes), dtype=tc.float)
            self_coeff = 0
            for conn in conns:
                for i in range(len(nodes)):
                    if nodes[i] in conn and nodes[i] != node_name:
                        eqn[i] = -1000
                        self_coeff += 1000
            # Now set the node itself to be the sum of the connection weights/coeffs
            for i in range(len(nodes)):
                if nodes[i] == node_name:
                    eqn[i] = self_coeff

        # Resistor voltage nodes have a special relation to the node on the other side of the resistor
        case 'res':
            eqn = tc.zeros(len(nodes), dtype=tc.float)
            self_coeff = 0
            for conn in conns:
                for i in range(len(nodes)):
                    if nodes[i] in conn and nodes[i] != node_name:
                        eqn[i] = -1000
                        self_coeff += 1000
            # Now add the connection to the other side of the resistor
            for i in range(len(nodes)):
                if comp_name in nodes[i] and node_name != nodes[i]:
                    eqn[i] = -(1 / prms)
                    self_coeff += (1 / prms)
            # Now set the node itself to be the sum of the connection weights/coeffs
            for i in range(len(nodes)):
                if nodes[i] == node_name:
                    eqn[i] = self_coeff

        case _:
            raise Exception(f"Component type {comp_type} not yet supported.")

    return eqn





class Component:
    def __init__(self):
        pass
