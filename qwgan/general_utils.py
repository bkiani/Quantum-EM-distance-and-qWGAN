import itertools
import numpy as np
import config
import math

if config.interface == 'torch':
	import torch
elif config.interface == 'tf':
	import tensorflow as tf

def enumerate_pauli_strings(n = 3, k = 2, return_as_strings = True):
	# may not be efficient for very large n and k
	qubit_combos = list(itertools.combinations(list(range(n)), k))
	Paulis = ['I','X','Y','Z']
	Pauli_list = ['I','X','Y','Z']
	for i in range(k-1):
		Pauli_list = [a+b for a in Paulis for b in Pauli_list]

	strings = set([get_pauli_string(n,i,j) for i in Pauli_list for j in qubit_combos])
	strings.remove('I'*n)
	output = list(strings)
	if return_as_strings:
		return output
	else:
		output = np.char.asarray([list(i) for i in output])
		qubit_applied = [np.where(np.char.not_equal(string_i, 'I'))[0].tolist() for string_i in output]
		Paulis = [output[i][qubit_applied[i]].tolist() for i in range(output.shape[0])]
		return qubit_applied, Paulis, list(strings)

def get_pauli_string(n,pauli,qubit_nums):
	out = list('I'*n)
	for i in range(len(qubit_nums)):
		out[qubit_nums[i]] = pauli[i]
	return ''.join(out)


def bool_operators_on_qubits(string_list):
	A = np.zeros([len(string_list[0]),len(string_list)]).astype(np.int)
	for i,string_i in enumerate(string_list):
		active_qubits = np.where(np.asarray(list(string_i)) != 'I')
		# print(active_qubits)
		A[active_qubits, i] = 1

	return A

def get_random_states(n_qubits = 3, rank = 1):
	N = 2**n_qubits
	states = []

	for i in range(rank):
		new_state = np.zeros((N,1)).astype(np.complex128)
		new_state[:,0].imag = np.random.normal(size=N)
		new_state[:,0].real = np.random.normal(size=N)
		new_state = new_state / np.sqrt(np.sum(np.abs(new_state)*np.abs(new_state)))
		states.append(new_state)

	return states

def states_to_density_mat(states, probs):
	output = np.zeros((len(states[0]),len(states[0]))).astype(np.complex128)
	for state,prob in zip(states,probs):
		state = state.reshape(-1,1)
		output += state@state.conj().T*prob
	return output


def trace_distance(d1,d2):
	s = np.linalg.svd(d1-d2,compute_uv=False,hermitian=True)
	return np.sum(s)


def get_random_normal_params(n, format = 'torch', **kwargs):
	if isinstance(n, int):
		n = [n]
	if format == 'torch':
		output = torch.empty(n, requires_grad = True, device = config.torch_device)
		with torch.no_grad():
			return output.normal_(**kwargs)
	elif format == 'tf':
		return tf.Variable(tf.convert_to_tensor(np.random.normal(size = n, **kwargs)))
	else:
		return np.rand.normal(size = n, **kwargs)


def pure_state_fidelity(state_1, state_2):
	overlap = np.sum(state_1*state_2.conj())
	return (overlap*overlap.conj()).real


def butterfly_indices(dim):
    # ### Only works for powers of 2 as input for dim

    # initialize a multi-dimensional list
    auxList = [[] for i in range(int(math.log(dim, 2)))]
    outList = [[] for i in range(int(math.log(dim, 2)))]

    # create auxList featuring the correct indices -> auxList: [[0, 1, 2, 3], [0, 2, 1, 3]]
    for ind in range(int(math.log(dim, 2))):
        currNum = 0
        stepSize = 2**ind
        auxList[ind] = [currNum]
        for indCyc in range(dim-1):
            currNum += stepSize
            if (currNum >= dim):
                currNum = currNum - dim + 1
            auxList[ind].append(currNum)

    # Convert auxList into the correct formatting -> outList:[[[0, 1], [2, 3]], [[0, 2], [1, 3]]]
    for ind in range(int(math.log(dim, 2))):
        for indCyc in range(int(dim/2)):
            outList[ind].append([auxList[ind][2*indCyc], auxList[ind][2*indCyc+1]])

    return outList

if __name__ == '__main__':
	s = enumerate_pauli_strings(3,2)
	print(s)
	s2 = enumerate_pauli_strings(3,2,False)
	print(s2)

	print(butterfly_indices(8))