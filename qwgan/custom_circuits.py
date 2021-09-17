import pennylane as qml
from pennylane import qaoa
import numpy as np
from general_utils import butterfly_indices
import math
from itertools import combinations
from random import choices


def get_mixer_hamiltonian(n):
	coeffs = [1. for i in range(n)]
	obs = [qml.PauliX(i) for i in range(n)]
	return qml.Hamiltonian(coeffs, obs)

def get_all_Z_hamiltonian(n):
	coeffs = [1. for i in range(n)]
	obs = [qml.PauliZ(i) for i in range(n)]
	return qml.Hamiltonian(coeffs, obs)	

def get_neighbor_ZZ_hamiltonian(n):
	coeffs = [1. for i in range(n)]
	obs = [qml.PauliZ(i)@qml.PauliZ((i+1)%n) for i in range(n)]
	return qml.Hamiltonian(coeffs, obs)	

def get_random_chain_hamiltonian(n):
	coeffs = np.random.normal(size=(n))
	obs = [qml.PauliZ(i)@qml.PauliZ((i+1)%n) for i in range(n)]
	return qml.Hamiltonian(coeffs, obs)	


def get_random_ZZ_hamiltonian(n, r_couples = 1.):
	n_couples = int(r_couples*n)

	coeffs = np.random.normal(size = (n+n_couples))

	combos = list(combinations(list(range(n)), 2))
	combos = choices(combos,k= n_couples)

	obs = [qml.PauliZ(i) for i in range(n)]
	obs += [qml.PauliZ(i[0])@qml.PauliZ(i[1]) for i in combos]
	return qml.Hamiltonian(coeffs, obs)	

def get_random_pairwise_ZZ_hamiltonian(n, p = 0.5, randomize = 'uniform'):
	obs = []

	if randomize == 'normal':
		coeffs = np.random.normal(size = (2*n))
	else:
		coeffs = np.random.rand(2*n)

	while len(obs) == 0:
		for i in range(n):
			if np.random.rand()<p:
				obs.append(qml.PauliZ(i))
			if np.random.rand()<p:
				obs.append(qml.PauliZ(i)@qml.PauliZ((i+1)%n))

	return qml.Hamiltonian(coeffs[:len(obs)], obs)	




def get_product_ZZ_hamiltonian(n ):
	n_half1 = int(n/2)
	n_half2 = n - n_half1
	ns_1 = choices(list(range(n)),k= n_half1)
	ns_2 = [i for i in range(n) if i not in ns_1]

	coeffs = np.random.normal(size = (4))

	obs = []
	all_coeffs = []
	# single Z for first half
	obs += [qml.PauliZ(i) for i in ns_1]
	all_coeffs += [coeffs[0] for i in ns_1]
	# single Z for second half
	obs += [qml.PauliZ(i) for i in ns_2]
	all_coeffs += [coeffs[1] for i in ns_2]
	# pairwise Z for first half
	if len(ns_1) == 2:
		obs += [qml.PauliZ(ns_1[0])@qml.PauliZ(ns_1[1])]
		all_coeffs += [coeffs[2]]
	else:
		obs += [qml.PauliZ(i)@qml.PauliZ((i+1) % len(ns_1)) for i in ns_1]
		all_coeffs += [coeffs[2] for i in ns_1]
	# pairwise Z for second half
	if len(ns_2) == 2:
		obs += [qml.PauliZ(ns_2[0])@qml.PauliZ(ns_2[1])]
		all_coeffs += [coeffs[3]]
	else:
		obs += [qml.PauliZ(i)@qml.PauliZ((i+1) % len(ns_2)) for i in ns_2]
		all_coeffs += [coeffs[3] for i in ns_2]

	print(ns_1)
	print(ns_2)
	print(obs)

	return qml.Hamiltonian(all_coeffs, obs)	






def qaoa_random_circuit(n, mixer_h = None, cost_Z = None, return_n_params = False, depth = 2):
	if mixer_h is None:
		mixer_h = get_mixer_hamiltonian(n)
	if cost_Z is None:
		cost_Z = get_random_ZZ_hamiltonian(n)

	def qaoa_layer(gamma):
		# qaoa.cost_layer(gamma[0], cost_Z1)
		qaoa.cost_layer(gamma[0], cost_Z)
		qaoa.mixer_layer(gamma[1], mixer_h)

	def qml_fun(params,  **kwargs):
		for i in range(n):
			qml.Hadamard(wires = i)
		for i in range(depth):
			qaoa_layer(params[i])
		# qml.layer(qaoa_layer, depth, params)

	if return_n_params:
		return qml_fun, [depth,2]
	else:
		return qml_fun





def qaoa_translation_circuit(n, return_n_params = False, depth = 2):
	mixer_h = get_mixer_hamiltonian(n)
	# cost_Z1 = get_all_Z_hamiltonian(n)
	cost_Z2 = get_neighbor_ZZ_hamiltonian(n)

	def qaoa_layer(gamma):
		# qaoa.cost_layer(gamma[0], cost_Z1)
		qaoa.cost_layer(gamma[0], cost_Z2)
		qaoa.mixer_layer(gamma[1], mixer_h)

	def qml_fun(params,  **kwargs):
		for i in range(n):
			qml.Hadamard(wires = i)
		for i in range(depth):
			qaoa_layer(params[i])
		# qml.layer(qaoa_layer, depth, params)

	if return_n_params:
		return qml_fun, [depth,2]
	else:
		return qml_fun



def single_qubit_rotations(n, return_n_params = False):
	def qml_fun(params, **kwargs):
		for i in range(n):
			qml.RX(params[i*3], wires = i)
			qml.RY(params[i*3+1], wires = i)
			qml.RZ(params[i*3+2], wires = i)
	
	if return_n_params:
		return qml_fun, n*3
	else:
		return qml_fun

def GHZ_constructor(n, return_n_params = False):
	def qml_fun(params, **kwargs):
		qml.RX(params[0], wires = 0)
		qml.RY(params[1], wires = 0)
		qml.RZ(params[2], wires = 0)
		for i in range(n-1):
			qml.CRX(params[3+i*3+0], wires = [i,i+1])
			qml.CRY(params[3+i*3+1], wires = [i,i+1])
			qml.CRZ(params[3+i*3+2], wires = [i,i+1])

	if return_n_params:
		return qml_fun, n*3
	else:
		return qml_fun


def GHZ_constructor_simple(n, return_n_params = False):
	def qml_fun(params, **kwargs):
		qml.RX(params[0], wires = 0)
		qml.RY(params[1], wires = 0)
		qml.RZ(params[2], wires = 0)
		for i in range(n-1):
			qml.CRX(params[3+i], wires = [i,i+1])

	if return_n_params:
		return qml_fun, n+2
	else:
		return qml_fun


def butterfly_Pauli_constructor(n, depth = 1, return_n_params = True):
	# Note, n must be a power of 2
	# depth indicates number of times circuit is repeated

	def qml_fun(params, **kwargs):
		count_param = 0
		for i in range(depth):
			for layer_j,indices_j in enumerate(indices):

				# first, apply single qubit rotations
				for k in range(n):
					qml.RX(params[count_param+0], wires = k)
					qml.RY(params[count_param+1], wires = k)
					qml.RZ(params[count_param+2], wires = k)	
					count_param += 3

				for next_index in indices_j:
					qml.CRX(params[count_param+0], wires = [next_index[0],next_index[1]])
					qml.CRY(params[count_param+1], wires = [next_index[0],next_index[1]])
					qml.CRZ(params[count_param+2], wires = [next_index[0],next_index[1]])
					count_param += 3

	indices = butterfly_indices(n)

	if return_n_params:
		return qml_fun, int(depth*len(indices)*3*(n+n/2))
	else:
		return qml_fun


def butterfly_Pauli_constructor_simple(n, depth = 1, return_n_params = True):
	# Note, n must be a power of 2
	# depth indicates number of times circuit is repeated

	def qml_fun(params, **kwargs):
		count_param = 0
		for i in range(depth):
			for layer_j,indices_j in enumerate(indices):

				# first, apply single qubit rotations
				for k in range(n):
					qml.RX(params[count_param+0], wires = k)
					# qml.RY(params[count_param+1], wires = k)
					# qml.RZ(params[count_param+2], wires = k)	
					count_param += 1

				for next_index in indices_j:
					qml.CRX(params[count_param+0], wires = [next_index[0],next_index[1]])
					# qml.CRY(params[count_param+1], wires = [next_index[0],next_index[1]])
					# qml.CRZ(params[count_param+2], wires = [next_index[0],next_index[1]])
					count_param += 1

	indices = butterfly_indices(n)

	if return_n_params:
		return qml_fun, int(depth*len(indices)*(n+n/2))
	else:
		return qml_fun


def generic_mixing_circuit(n, depth = 1, return_n_params = True):

	def qml_fun(params, **kwargs):
		count_param = 0
		for i in range(depth):
			for layer_i in range(n):

				# first, apply single qubit rotations
				for k in range(n):
					qml.Rot(params[count_param+0], params[count_param+1], params[count_param+2], wires = k)
					count_param += 3

				for k in range(int(n/2)):
					qml.CRY(params[count_param+0], wires = [(2*k+layer_i)%n,(2*k+layer_i+1)%n])
					qml.CRY(params[count_param+0], wires = [(2*k+layer_i+1)%n,(2*k+layer_i+2)%n])
					count_param += 2

	if return_n_params:
		return qml_fun, int(depth*(4*n*n))
	else:
		return qml_fun

def simple_mixing_circuit(n, depth = 1, return_n_params = True):

	def qml_fun(params, **kwargs):
		count_param = 0
		for i in range(depth):
			for layer_i in range(n):

				# first, apply single qubit rotations
				for k in range(n):
					qml.RY(params[count_param], wires = k)
					count_param += 1

				for k in range(int(n/2)):
					qml.CRY(params[count_param+0], wires = [(2*k+layer_i)%n,(2*k+layer_i+1)%n])
					qml.CRY(params[count_param+0], wires = [(2*k+layer_i+1)%n,(2*k+layer_i+2)%n])
					count_param += 2

	if return_n_params:
		return qml_fun, int(depth*(2*n*n))
	else:
		return qml_fun


def random_mixing_circuit(n, depth = 1, return_n_params = True):

	def qml_fun(params, **kwargs):
		count_param = 0
		for i in range(depth):
			# first, apply single qubit rotations
			for k in range(n):
				qml.RY(params[count_param], wires = k)
				count_param += 1

			# first shuffle
			for k in interactions[2*i]:
				qml.CRY(params[count_param+0], wires = [k[0],k[1]])
				count_param += 1

			# second shuffle
			for k in interactions[2*i+1]:
				qml.CRY(params[count_param+0], wires = [k[0],k[1]])
				count_param += 1

	# interactions = [np.random.shuffle(np.arange(n)).reshape(int(n/2),2) for i in range(2*depth)]
	interactions = []
	for _ in range(2*depth):
		qubits = np.arange(n).astype(np.int)
		np.random.shuffle(qubits)
		interactions.append(qubits.reshape(-1,2))


	if return_n_params:
		return qml_fun, int(depth*(2*n))
	else:
		return qml_fun


def bp_circuit(n, depth = 1, return_n_params = True):

	def qml_fun(params, **kwargs):
		count_param = 0
		for i in range(depth):
			# first, apply single qubit rotations
			for k in range(n):
				qml.RY(params[count_param], wires = k)
				count_param += 1

			# first mixing gates
			for k in range(int(n/2)):
				qml.MultiRZ(params[count_param], wires = [int(2*k),int(2*k+1)])
				count_param += 1

			# second single qubit rotations
			for k in range(n):
				qml.RY(params[count_param], wires = k)
				count_param += 1

			# second mixing gates
			for k in range(int(n/2)):
				qml.MultiRZ(params[count_param], wires = [int(2*k+1),int((2*k+2)%n)])
				count_param += 1

	if return_n_params:
		return qml_fun, int(depth*(2*n+2*int(n/2)))
	else:
		return qml_fun




def draw_circuit(fun_name, n):
	dev = qml.device('default.qubit', wires = n)
	call_fun, n_params = fun_name(n, return_n_params = True)
	# circuit = qml.QNode(qml_fun, dev)
	if isinstance(n_params, int):
		n_params = [n_params]
	qnodes = qml.map(call_fun, [qml.Identity(0)], dev, measure="expval")
	measurements = qnodes( np.arange(np.prod(np.asarray(n_params)), step = 1).reshape(n_params) )
	# print(qnodes.qnodes[0].print_applied())
	print(qnodes.qnodes[0].draw(charset = 'ascii'))





if __name__ == '__main__':
	print('QAOA')
	draw_circuit(qaoa_translation_circuit, 4)
	print('Butterfly Circuit:')
	draw_circuit(butterfly_Pauli_constructor_simple,4)
	print()
	print()
	print('GHZ Circuit:')
	draw_circuit(GHZ_constructor,4)
	print()
	print()
	print('GHZ Circuit:')
	draw_circuit(generic_mixing_circuit,4)
	print()
	print()
	print('BP Circuit')
	draw_circuit(bp_circuit, 5)

	# ghz_circ = GHZ_constructor_simple(5)
	# draw_circuit(GHZ_constructor_simple,4)

