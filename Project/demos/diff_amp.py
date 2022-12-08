"""
DC circuit for assembly with two voltage sources used as inputs to an op amp differential amplifier configuration.
"""

import debugbuddy


def run_demo():
    print('Running differential amplifier faulty circuit demo!')
    # Define the intended circuit design
    components = {'v_in': ['v1', 'v2', 'gnd', 'vcc'], 'v_out': ['vo'],
                  'res': ['r1', 'r2', 'r3', 'r4', 'rl'], 'opamp5': ['u1']}
    prms = {'r1-r': 1000, 'r2-r': 1000, 'r3-r': 4000, 'r4-r': 4000, 'rl-r': 20000,
            'u1-g': 10000, 'u1-ri': 1000000, 'u1-ro': 50}
    correct_conns = [{'v1', 'r1.1'}, {'r1.2', 'u1.-'}, {'v2', 'r2.1'}, {'r2.2', 'u1.+'},
                     {'u1.-', 'r3.1'}, {'r3.2', 'vo'}, {'u1.+', 'r4.1'}, {'r4.2', 'gnd'},
                     {'gnd', 'rl.1'}, {'vo', 'rl.2'}, {'vo', 'u1.o'}, {'u1.vcc', 'vcc'}, {'u1.vee', 'gnd'}]
    # For now circuit is actually assembled correctly
    faulty_conns = [{'v1', 'r1.1'}, {'r1.2', 'u1.-'}, {'v2', 'r2.1'}, {'r2.2', 'u1.+'},
                    {'u1.-', 'r3.1'}, {'r3.2', 'vo'}, {'u1.+', 'r4.1'}, {'r4.2', 'gnd'},
                    {'gnd', 'rl.1'}, {'vo', 'rl.2'}, {'vo', 'u1.o'}, {'u1.vcc', 'vcc'}, {'u1.vee', 'gnd'}]

    circ = debugbuddy.FaultyCircuit(components, faulty_conns, correct_conns, prms)
    outs = circ.simulate_test([0.4, 0.6, 0, 1])
    print(outs)

    debugbuddy.guided_debug(circ)


if __name__ == '__main__':
    run_demo()
