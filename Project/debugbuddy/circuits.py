"""Defines the BOED-compatible linear circuit solver for use in circuit debug."""

import pyro
import torch as tc

from .components import *

__all__ = ['FaultyCircuit']


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

    def get_kcl_eqn(self, mode='sim', comp_prms=None):
        if mode == 'sim':
            return get_eqn(self.name, self.parent_comp, self.type, self.nodes, self.conns, comp_prms)


class FaultyCircuit:
    def __init__(self, components, faulty_conns, intended_conns, prms):
        self.components = components
        self.nodes = self._construct_nodes(components, faulty_conns)
        self.priors = intended_conns
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
                    nodes.append(CircuitNode(comp_name + '-1', comp_name, comp_type, ordered, conns))
                    nodes.append(CircuitNode(comp_name + '-2', comp_name, comp_type, ordered, conns))
                else:
                    nodes.append(CircuitNode(comp_name, comp_name, comp_type, ordered, conns))
        return nodes

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
            with pyro.plate_stack('weigh-plate', test_ins.shape[:-1]):
                # Setup KCL node voltage equations

                # Solve the system of equations to get the node voltages
                node_voltages = tc.linalg.solve()
                # Only return the measured node voltages
                return node_voltages # FIXME

        return fault_mdl
