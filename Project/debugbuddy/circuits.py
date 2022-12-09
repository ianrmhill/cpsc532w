"""Defines the BOED-compatible linear circuit solver for use in circuit debug."""

import torch as tc
import pyro
import pyro.distributions as dist

from .components import *

__all__ = ['FaultyCircuit']

MAX_RES = 1e6


def prm_dist(p1, p2, p3, p4):
    return 0.5 * dist.Beta(p1, p2).sample() + 0.5 * dist.Normal(p3, p4).sample()


def conn_dist(p1, p2, p3, p4):
    return 0.5 * dist.Beta(p1, p2).sample() + 0.5 * dist.Normal(p3, p4).sample()


class CircuitNode:
    def __init__(self, name, comp, type, all_nodes, connections):
        self.name = name
        self.parent_comp = comp
        self.type = type
        self.nodes = all_nodes
        self.conns = []
        for conn in connections:
            if name in conn:
                self.conns.append(conn)

    def get_kcl_eqn(self, mode='sim', comp_prms=None, edge_states=None, batch_shape=None):
        if mode == 'sim':
            return get_eqn(self.name, self.parent_comp, self.type, self.nodes, self.conns, comp_prms)
        elif mode == 'predict':
            return get_pred_eqn(self.name, self.parent_comp, self.type, self.nodes, edge_states, comp_prms, batch_shape)
        else:
            raise Exception(f"Invalid KCL equation construction mode: {mode}")


class FaultyCircuit:
    def __init__(self, components, faulty_conns, intended_conns, prms):
        self.components = components
        self.nodes = self._construct_nodes(components, faulty_conns)
        self.edges = self.get_edges()
        self.priors = self._construct_priors(intended_conns, prms)
        self.comp_prms = prms

    @staticmethod
    def _construct_nodes(components, conns):
        # First need to determine what nodes will exist to know the length of our connection listings
        ordered = []
        for comp_type in components:
            for comp_name in components[comp_type]:
                ordered.extend(get_comp_nodes(comp_name, comp_type))

        # Now build the node objects with the prior knowledge of which nodes will exist
        nodes = []
        for comp_type in components:
            for comp_name in components[comp_type]:
                if comp_type == 'res':
                    nodes.append(CircuitNode(comp_name + '.1', comp_name, comp_type, ordered, conns))
                    nodes.append(CircuitNode(comp_name + '.2', comp_name, comp_type, ordered, conns))
                elif comp_type == 'opamp5':
                    nodes.append(CircuitNode(comp_name + '.-', comp_name, comp_type, ordered, conns))
                    nodes.append(CircuitNode(comp_name + '.+', comp_name, comp_type, ordered, conns))
                    nodes.append(CircuitNode(comp_name + '.o', comp_name, comp_type, ordered, conns))
                    nodes.append(CircuitNode(comp_name + '.vcc', comp_name, comp_type, ordered, conns))
                    nodes.append(CircuitNode(comp_name + '.vee', comp_name, comp_type, ordered, conns))
                elif comp_type == 'opamp3':
                    nodes.append(CircuitNode(comp_name + '.-', comp_name, comp_type, ordered, conns))
                    nodes.append(CircuitNode(comp_name + '.+', comp_name, comp_type, ordered, conns))
                    nodes.append(CircuitNode(comp_name + '.o', comp_name, comp_type, ordered, conns))
                else:
                    nodes.append(CircuitNode(comp_name, comp_name, comp_type, ordered, conns))
        return nodes

    def _construct_priors(self, expected_conns, prms):
        priors = {}
        for prm in prms:
            priors[prm] = [prms[prm], prms[prm] * 0.05]
        for edge in self.edges:
            edge_name = str(sorted(tuple(edge)))
            if edge in expected_conns:
                priors[edge_name] = 0.9
            else:
                priors[edge_name] = 0.1
        return priors

    def get_obs_lbls(self):
        obs_lbls = []
        for node in self.nodes:
            if node.type == 'v_out':
                obs_lbls.append(node.name)
        return obs_lbls

    def get_latent_lbls(self):
        ltnt_lbls = []
        for prm in self.comp_prms:
            ltnt_lbls.append(prm)
        for node1 in self.nodes:
            for node2 in self.nodes:
                edge_name = str(sorted(tuple({node1.name, node2.name})))
                if node1.name != node2.name and f"{edge_name}-r" not in ltnt_lbls:
                    ltnt_lbls.append(f"{edge_name}-r")
        return ltnt_lbls

    def get_edges(self):
        edges = []
        for node1 in self.nodes:
            for node2 in self.nodes:
                if node1.name != node2.name and {node1.name, node2.name} not in edges:
                    edges.append({node1.name, node2.name})
        return edges

    def kcl_solver(self, v_ins, forced_nodes=None):
        # Default to no enforced node voltage overrides
        if not forced_nodes:
            forced_nodes = {}
        # For our KCL equations we will always accumulate all terms on one side of the equality, thus B is a 0 vector
        # except for at fixed input voltage nodes
        b = tc.zeros(len(self.nodes))
        j = 0
        for i, node in enumerate(self.nodes):
            if node.type == 'v_in':
                b[i] = v_ins[j]
                j += 1
            elif node.name in forced_nodes:
                b[i] = forced_nodes[node.name]
        # Check that the supplied number of input voltages was correct
        if len(v_ins) != j:
            raise Exception(f"Incorrect number of input voltages provided. Provided: {len(v_ins)}. Needed: {j}")

        # Build up set of KCL equations for all the terminal nodes
        a_list = []
        for i, node in enumerate(self.nodes):
            # First check that the node voltage isn't being forced to override
            if node.type == 'v_in' or node.name in forced_nodes:
                eqn = tc.zeros(len(self.nodes), dtype=tc.float)
                eqn[i] = 1
                a_list.append(eqn)
            else:
                if node.type == 'res':
                    prms = self.comp_prms[f"{node.parent_comp}-r"]
                elif node.type == 'opamp3' or node.type == 'opamp5':
                    prms = [self.comp_prms[f"{node.parent_comp}-g"],
                            self.comp_prms[f"{node.parent_comp}-ri"], self.comp_prms[f"{node.parent_comp}-ro"]]
                else:
                    prms = None
                a_list.append(node.get_kcl_eqn('sim', prms))
        a = tc.stack(a_list)

        # Now solve the system of equations
        v = tc.linalg.solve(a, b)
        return v

    def simulate_test(self, v_ins):
        # Solve the linear circuit given the voltage inputs
        v = self.kcl_solver(v_ins)

        # Non-linear circuit effect handling!!! For now just handling op amp power rail saturation
        for i, node in enumerate(self.nodes):
            if node.type == 'opamp5' and '.o' in node.name:
                v_min, v_max = None, None
                for j, node2 in enumerate(self.nodes):
                    if node2.parent_comp == node.parent_comp:
                        if '.vcc' in node2.name:
                            v_max = v[j]
                        elif '.vee' in node2.name:
                            v_min = v[j]
                if v_min is not None and v[i] < v_min:
                    v = self.kcl_solver(v_ins, {node.name: v_min})
                elif v_max is not None and v[i] > v_max:
                    v = self.kcl_solver(v_ins, {node.name: v_max})

        # Finally, return the observed output voltages
        out_list = []
        for i, node in enumerate(self.nodes):
            if node.type == 'v_out':
                out_list.append(v[i])
        return tc.tensor(out_list)

    def inf_kcl_solver(self, v_ins, prms, shorts, forced_nodes=None):
        # Default to no enforced node voltage overrides
        if not forced_nodes:
            forced_nodes = {}
        # Setup fixed voltage vector
        b = tc.zeros((*v_ins.shape[:-1], len(self.nodes)))
        j = 0
        for i, node in enumerate(self.nodes):
            if node.type == 'v_in':
                b[..., i] = v_ins[..., j]
                j += 1
            elif node.name in forced_nodes:
                b[..., i] = forced_nodes[node.name]

        # Setup KCL node voltage equations
        a_list = []
        for i, node in enumerate(self.nodes):
            if node.type == 'v_in' or node.name in forced_nodes:
                eqn = tc.zeros((*v_ins.shape[:-1], len(self.nodes)), dtype=tc.float)
                eqn[..., i] = 1
                a_list.append(eqn)
            else:
                if node.type == 'res':
                    kcl_prms = prms[f"{node.parent_comp}-r"]
                elif node.type == 'opamp3' or node.type == 'opamp5':
                    kcl_prms = [prms[f"{node.parent_comp}-g"],
                                prms[f"{node.parent_comp}-ri"], prms[f"{node.parent_comp}-ro"]]
                else:
                    kcl_prms = None
                a_list.append(node.get_kcl_eqn('predict', kcl_prms, shorts, v_ins.shape[:-1]))
        a = tc.stack(a_list, -2)

        # Solve the system of equations to get the node voltages
        v = tc.linalg.solve(a, b)
        return v

    def gen_fault_mdl(self, beliefs=None):
        # If no beliefs are provided it means we are using the initial intended connections for our priors
        if not beliefs:
            beliefs = self.priors

        def fault_mdl(test_ins):
            with pyro.plate_stack('iso-plate', test_ins.shape[:-1]):
                # Sample all our latent parameters
                prms = {}
                for comp in self.comp_prms:
                    prms[comp] = pyro.sample(f"{comp}-r", dist.Normal(*beliefs[comp]))
                shorts = {}
                for edge in self.edges:
                    edge_name = str(sorted(tuple(edge)))
                    shorts[edge_name] = pyro.sample(
                        f"{edge_name}-r", dist.Categorical(tc.tensor([beliefs[edge_name], 1 - beliefs[edge_name]])))

                v = self.inf_kcl_solver(test_ins, prms, shorts)

                # Non-linear circuit effect handling!!! For now just handling op amp power rail saturation
                for i, node in enumerate(self.nodes):
                    if node.type == 'opamp5' and '.o' in node.name:
                        v_min, v_max = None, None
                        for j, node2 in enumerate(self.nodes):
                            if node2.parent_comp == node.parent_comp:
                                if '.vcc' in node2.name:
                                    v_max = v[..., j]
                                elif '.vee' in node2.name:
                                    v_min = v[..., j]
                        # Because the control flow logic of the KCL solver is at the batch level, have to individually
                        # rerun cases where the op amp limits are exceeded, yikes, it's too slow
                        if v_min is not None:
                            batch_dims = test_ins.shape[:-1]
                            if len(batch_dims) == 1:
                                for n in range(batch_dims[0]):
                                    if v[n, i] < v_min[n]:
                                        v[n, :] = self.inf_kcl_solver(
                                            test_ins[n, :], {prm: prms[prm][n] for prm in prms},
                                            {edge: shorts[edge][n] for edge in shorts},
                                            {node.name: v_min[n]})
                            elif len(batch_dims) == 2:
                                for n in range(batch_dims[0]):
                                    for m in range(batch_dims[1]):
                                        if v[n, m, i] < v_min[n, m]:
                                            v[n, m, :] = self.inf_kcl_solver(
                                                test_ins[n, m, :], {prm: prms[prm][n, m] for prm in prms},
                                                {edge: shorts[edge][n, m] for edge in shorts},
                                                {node.name: v_min[n, m]})
                        elif v_max is not None and v[i] > v_max:
                            v = self.inf_kcl_solver(test_ins, prms, shorts, {node.name: v_max})

                # Only return the measured node voltages
                out_list = []
                for i, node in enumerate(self.nodes):
                    if node.type == 'v_out':
                        out_list.append(pyro.sample(f"{node.name}", dist.Normal(v[..., i], 0.002)))
                outs = tc.stack(out_list, -1)
                return outs

        return fault_mdl
