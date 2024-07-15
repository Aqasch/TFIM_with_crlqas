from itertools import product
import numpy as np
from qiskit import QuantumCircuit

def dictionary_of_actions(num_qubits):
    dictionary = dict()
    i = 0
    for c, x in product(range(num_qubits),
                        range(1, num_qubits)):
        dictionary[i] =  [c, x, num_qubits, 0]
        i += 1
    for r, h in product(range(num_qubits),
                           range(1, 4)):
        dictionary[i] = [num_qubits, 0, r, h]
        i += 1
    return dictionary

def make_circuit_qiskit(action, qubits):
    ctrl = action[0]
    targ = (action[0] + action[1]) % qubits
    rot_qubit = action[2]
    rot_axis = action[3]
#     print(1)
    if ctrl < qubits:
        circuit.cx([ctrl], [targ])
    if rot_qubit < qubits:
        if rot_axis == 1:
            circuit.rx(0, rot_qubit) # TODO: make a function and take angles
        elif rot_axis == 2:
            circuit.ry(0, rot_qubit)
        elif rot_axis == 3:
            circuit.rz(0, rot_qubit)
    
    return circuit


seed = 1234
episodes = 1251

# for seed in range(1, seed+1):
for seed in [seed]:
    print(f'SEED: {seed}')
    # data = np.load(f'results/finalize/lbmt_cobyla_H24q0p35_kaqn_seed{seed}/summary_{seed}.npy',allow_pickle=True)[()]
    # data = np.load(f'results/finalize/lbmt_cobyla_H24q0p35/summary_{seed}.npy',allow_pickle=True)[()]
    # data = np.load(f'results/finalize/lbmt_cobyla_LiH4q3p4_wo_rand_halt_kan_neuron30x3_seed{seed}/summary_{seed}.npy',allow_pickle=True)[()]
    data = np.load(f'results/finalize/test/summary_{seed}.npy',allow_pickle=True)[()]
    
    error_list = []
    time_list = []
    succ_ep_list = []
    actions_len_list = []
    for ep in range(0, episodes):
        #H2
        # err = data['train'][ep]['errors'][-1]-1.99009719+1.136189454187329
        
        #LiH
        err = data['train'][ep]['errors'][-1]-3+2.2360679774997894
        error_list.append(err)

        # print(err)
        actions_len_list.append(len(data['train'][ep]['actions']))
        if err <=0.0016:
            succ_ep_list.append(ep)


    min_err_ep = error_list.index(np.min(error_list))
    print(min_err_ep, np.min(error_list))

    qubits = 2
    twoq_gate_list, oneq_gate_list, gate_num_list, depth_list, circ_list = [], [], [], [], []
    for succ_ep in [min_err_ep]:
    # for succ_ep in succ_ep_list:

        actions = data['train'][succ_ep]['actions']
        circuit = QuantumCircuit(qubits)
        for a in actions:
            action = dictionary_of_actions(qubits)[a]
            # print(action)
            final_circuit = make_circuit_qiskit(action, qubits)
        print(final_circuit)
        gate_info = final_circuit.count_ops()
        key_list = gate_info.keys()
        one_gate, two_gate = 0,0
        for k in key_list:
            if k == 'cx':
                two_gate += gate_info[k]
            else:
                one_gate += gate_info[k]
        circ_list.append(final_circuit)
        twoq_gate_list.append(two_gate)
        oneq_gate_list.append(one_gate)
        gate_num_list.append(one_gate+two_gate)        
        depth_list.append(final_circuit.depth())

    print('Minimum depth', np.min(depth_list))
    print('Minimum tot gate num', np.min(gate_num_list))
    print('Minimum 1q gate', np.min(oneq_gate_list))
    print('Minimum 2q gate', np.min(twoq_gate_list))
    print('-x-x-x-x-x-x-x-x-x-x-x-x-x-')
    print()