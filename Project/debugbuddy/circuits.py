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
            ltnt_lbls.append(f"{prm}-r")
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

    def simulate_test(self, v_ins):
        # For our KCL equations we will always accumulate all terms on one side of the equality, thus B is a 0 vector
        # except for at fixed input voltage nodes
        b = tc.zeros(len(self.nodes))
        j = 0
        for i, node in enumerate(self.nodes):
            if node.type == 'v_in':
                b[i] = v_ins[j]
                j += 1
        # Check that the supplied number of input voltages was correct
        if len(v_ins) != j:
            raise Exception(f"Incorrect number of input voltages provided. Provided: {len(v_ins)}. Needed: {j}")

        # Build up set of KCL equations for all the terminal nodes
        a_list = []
        for node in self.nodes:
            if node.type == 'res':
                prms = self.comp_prms[node.parent_comp]
            else:
                prms = None
            a_list.append(node.get_kcl_eqn('sim', prms))
        a = tc.stack(a_list)

        # Now solve the system of equations and return the output voltages
        v = tc.linalg.solve(a, b)
        out_list = []
        for i, node in enumerate(self.nodes):
            if node.type == 'v_out':
                out_list.append(v[i])
        return tc.tensor(out_list)

    def gen_fault_mdl(self, beliefs=None):
        # If no beliefs are provided it means we are using the initial intended connections for our priors
        if not beliefs:
            beliefs = self.priors

        def fault_mdl(test_ins):
            with pyro.plate_stack('iso-plate', test_ins.shape[:-1]):
                # Setup fixed voltage vector
                b = tc.zeros((*test_ins.shape[:-1], len(self.nodes)))
                j = 0
                # Masking may be the best solution here, just pre-gen the masks for different node types
                for i, node in enumerate(self.nodes):
                    if node.type == 'v_in':
                        b[..., i] = test_ins[..., j]
                        j += 1

                # Sample all our latent parameters
                prms = {}
                for comp in self.comp_prms:
                    prms[comp] = pyro.sample(f"{comp}-r", dist.Normal(*beliefs[comp]))
                shorts = {}
                for edge in self.edges:
                    edge_name = str(sorted(tuple(edge)))
                    shorts[edge_name] = pyro.sample(
                        f"{edge_name}-r", dist.Categorical(tc.tensor([beliefs[edge_name], 1 - beliefs[edge_name]])))

                # Setup KCL node voltage equations
                a_list = []
                for node in self.nodes:
                    kcl_prms = prms[node.parent_comp] if node.type == 'res' else None
                    a_list.append(node.get_kcl_eqn('predict', kcl_prms, shorts, test_ins.shape[:-1]))
                a = tc.stack(a_list, -2)

                # Solve the system of equations to get the node voltages
                v = tc.linalg.solve(a, b)
                # Only return the measured node voltages
                out_list = []
                for i, node in enumerate(self.nodes):
                    if node.type == 'v_out':
                        out_list.append(pyro.sample(f"{node.name}", dist.Normal(v[..., i], 0.002)))
                outs = tc.stack(out_list, -1)
                return outs

        return fault_mdl
