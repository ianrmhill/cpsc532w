
import torch as tc

__all__ = ['get_comp_nodes', 'get_eqn', 'get_pred_eqn']


def get_comp_nodes(comp_name, comp_type):
    match comp_type:
        case 'v_in': return [comp_name]
        case 'v_out': return [comp_name]
        case 'res': return [f"{comp_name}.1", f"{comp_name}.2"]
        case 'opamp3': return [f"{comp_name}.-", f"{comp_name}.+", f"{comp_name}.o"]
        case 'opamp5': return [f"{comp_name}.-", f"{comp_name}.+", f"{comp_name}.o",
                              f"{comp_name}.vcc", f"{comp_name}.vee"]
        case _: raise Exception(f"Unknown component type {comp_type}!")


def get_eqn(node_name, comp_name, comp_type, nodes, conns, prms):
    eqn = tc.zeros(len(nodes), dtype=tc.float)
    self_coeff = 0
    for conn in conns:
        for i in range(len(nodes)):
            if nodes[i] in conn and nodes[i] != node_name:
                eqn[i] += -1000
                self_coeff += 1000
    # Now add the sum of the short connections to the self node coefficient
    for i in range(len(nodes)):
        if nodes[i] == node_name:
            eqn[i] += self_coeff

    # Now handle special behaviour for terminals connected to components
    match comp_type:
        # Resistor voltage nodes have a special relation to the node on the other side of the resistor
        case 'res':
            # Add the influence from the connection to the other side of the resistor
            for i in range(len(nodes)):
                if comp_name in nodes[i]:
                    if nodes[i] == node_name:
                        eqn[i] += (1 / prms)
                    else:
                        eqn[i] += -(1 / prms)

        case 'opamp3' | 'opamp5':
            if '.-' in node_name or '.+' in node_name:
                # Add the resistive connection to the other input terminal
                for i in range(len(nodes)):
                    if comp_name in nodes[i] and ('.-' in nodes[i] or '.+' in nodes[i]):
                        if nodes[i] == node_name:
                            eqn[i] += (1 / prms[1])
                        else:
                            eqn[i] += -(1 / prms[1])
            elif '.o' in node_name:
                # Add the op amp gain influence
                for i in range(len(nodes)):
                    if comp_name in nodes[i]:
                        if nodes[i] == node_name:
                            eqn[i] += (1 / prms[2])
                        elif '.-' in nodes[i]:
                            eqn[i] += (prms[0] / prms[2])
                        elif '.+' in nodes[i]:
                            eqn[i] += -(prms[0] / prms[2])

    return eqn


def get_pred_eqn(node_name, comp_name, comp_type, nodes, edge_states, prms, batch_shape):
    match comp_type:
        # If input voltage, will just be equal to 1 times itself, no connections influence the value
        case 'v_in':
            eqn = tc.zeros((*batch_shape, len(nodes)), dtype=tc.float)
            for i in range(len(nodes)):
                if nodes[i] == node_name:
                    eqn[..., i] = 1

        # Output voltage nodes are the generic KCL case, no special behaviour needed
        case 'v_out':
            eqn = tc.zeros((*batch_shape, len(nodes)), dtype=tc.float)
            self_coeff = tc.zeros(batch_shape, dtype=tc.float)
            for i, node in enumerate(nodes):
                # Don't yet handle coefficients for the node itself
                if node != node_name:
                    edge_name = str(sorted(tuple({node, node_name})))
                    state = edge_states[edge_name]
                    eqn[..., i] = -1000 * state
                    self_coeff += 1000 * state
            # Now set the node itself to be the sum of the connection weights/coeffs
            for i in range(len(nodes)):
                if nodes[i] == node_name:
                    eqn[..., i] = self_coeff

        # Resistor voltage nodes have a special relation to the node on the other side of the resistor
        case 'res':
            eqn = tc.zeros((*batch_shape, len(nodes)), dtype=tc.float)
            self_coeff = tc.zeros(batch_shape, dtype=tc.float)
            for i, node in enumerate(nodes):
                # Don't yet handle coefficients for the node itself or the connection to the other side of the resistor
                if node != node_name and comp_name not in node:
                    edge_name = str(sorted(tuple({node, node_name})))
                    state = edge_states[edge_name]
                    eqn[..., i] = -1000 * state
                    self_coeff += 1000 * state
            # Now add the connection to the other side of the resistor, which is potentially two resistors in parallel
            for i in range(len(nodes)):
                # Identify the node that represents the other resistor terminal
                if comp_name in nodes[i] and node_name != nodes[i]:
                    edge_name = str(sorted(tuple({nodes[i], node_name})))
                    state = edge_states[edge_name]
                    # For now just set to be either the resistor resistance or a short, do full parallel once continuous
                    eqn[..., i] = tc.where(state == 1, -(1 / prms), -1000)
                    self_coeff += tc.where(state == 1, (1 / prms), 1000)
            # Now set the node itself to be the sum of the connection weights/coeffs
            for i in range(len(nodes)):
                if nodes[i] == node_name:
                    eqn[..., i] = self_coeff

        case _:
            raise Exception(f"Component type {comp_type} not yet supported.")

    return eqn
