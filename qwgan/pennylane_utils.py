import general_utils
import pennylane as qml
import numpy as np
import config

pauli_op_dict = {
					'X': qml.PauliX,
					'Y': qml.PauliY,
					'Z': qml.PauliZ
				}

pauli_op_list = [qml.Identity, qml.PauliX, qml.PauliY, qml.PauliZ]

def enumerate_pauli_qml_ops(n=3,k=2, return_map = True):
	qubits, strings, strings_full = general_utils.enumerate_pauli_strings(n,k,False)
	ops = []

	if return_map:
		A = np.zeros([n,len(strings)]).astype(np.int)
		for i in range(len(strings)):
			A[qubits[i],i] = 1

	for qubit, string in zip(qubits,strings):
		ops.append(get_pauli_op(qubit,string, n))

	if return_map:
		return ops, strings_full, A
	else:
		return ops


def get_random_poly_ops(n=3, ignore_set = set(), n_ops = 100, pauli_probs = [1/4]*4, return_map = True):
	ops = []
	op_strings = []
	n_ops_added = 0
	A = np.zeros([n,n_ops]).astype(np.int)
	if isinstance(ignore_set, list): 
		ignore_set.append('I'*n)
	else:
		ignore_set.add('I'*n)

	while n_ops_added < n_ops:
		pauli_digits = np.random.choice(4, size = (n, n_ops*2), replace = True, p = pauli_probs )
		pauli_letters = convert_pauli_digits_to_letters(pauli_digits)
		Atemp = pauli_digits > 0

		for i in range(n_ops*2):
			qubits = np.argwhere(Atemp[:,i] > 0).reshape(-1)
			current_letters = pauli_letters[i]
			if len(qubits) > 0 and current_letters not in ignore_set and current_letters not in op_strings and len(current_letters)==n:
				string = pauli_digits[qubits,i].reshape(-1)
				ops.append(get_pauli_op_from_ints(qubits, string, n))
				A[qubits,n_ops_added] = 1
				n_ops_added += 1
				op_strings.append(current_letters)
				if n_ops_added >= n_ops:
					break


	if return_map:
		return ops, op_strings, A
	else:
		return ops, op_strings

def convert_pauli_digits_to_letters(A):
	M = np.chararray(A.shape, unicode = True)
	M[A == 0] = "I"
	M[A == 1] = "X"
	M[A == 2] = "Y"
	M[A == 3] = "Z"
	output = np.chararray(M.shape[1], unicode = True)
	for i in range(A.shape[0]):
		output = np.char.add(output,M[i,:])
	return output.tolist()


def get_pauli_op(wires,paulis, n = None):
	ops = [pauli_op_dict[i](j) for i,j in zip(paulis, wires)]
	# tf interface has issues with broadcasting tensors of large sizes so need to add identities
	# if config.interface == 'tf':
	# 	# ops = ops + [qml.Identity(i) for i in range(n) if i not in wires]
	# 	ids = [i for i in range(n) if i not in wires]
	# 	ops = ops + [qml.Identity(wires = ids)]
	return qml.operation.Tensor(*ops)

def get_pauli_op_from_ints(wires, paulis, n = None):
	ops = [pauli_op_list[i](j) for i,j in zip(paulis, wires)]
	# if config.interface == 'tf':
		# ops = ops + [qml.Identity(i) for i in range(n) if i not in wires]
		# ids = [i for i in range(n) if i not in wires]
		# ops = ops + [qml.Identity(wires = ids)]
	return qml.operation.Tensor(*ops)

qpu = qml.device('default.qubit', wires=2)
@qml.qnode(qpu)
def op_as_given_state(params, state = None, wires = None, ops = None):
	qml.QubitStateVector(state, wires = wires)
	return [qml.expval(op) for op in ops]

if __name__ == '__main__':
	# op_test = [qml.PauliZ(0),qml.PauliY(1)]
	# # print(qml.PauliZ(0)@qml.PauliY(1))
	# sample_op = qml.operation.Tensor(*op_test)
	# # print(qml.operation.Tensor(*op_test))
	# # print(enumerate_pauli_qml_ops(2,2,True))

	# # state = np.asarray(np.zeros(4).astype(np.complex128))
	# # state[-1] = 1.j
	# # sample_state = op_as_given_state

	# # ops, A_map = enumerate_pauli_qml_ops(2,2)

	# # # qpu = qml.device('default.qubit', wires=2)
	# # # qnodes = qml.map(op_as_given_state, [sample_op], qpu, measure="expval")
	# # # print(qnodes({'s': state, 'q': [0,1]}))
	# # print(op_as_given_state([], state = state, wires = [0,1], ops = [ops[0]]))

	# summed_op = qml.operation.Tensor()
	# summed_op += 1.*op_test[0]
	# summed_op += 0.5*op_test[1]
	# print(0.5*op_test[0]+1.*op_test[1])


	print(enumerate_pauli_qml_ops(2,2))
	ops = get_random_poly_ops(5, ['XIIII','YIIII','ZIIII'], 100)
	# print(ops)