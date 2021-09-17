import pennylane as qml
from pennylane_utils import enumerate_pauli_qml_ops, get_random_poly_ops
from general_utils import pure_state_fidelity, trace_distance, states_to_density_mat
import config
import numpy as np
import cvxpy
import time
import pandas as pd
from pennylane.qnodes import PassthruQNode

if config.interface == 'torch':
	import torch
	from torch.nn.functional import relu, softmax


if config.interface == 'tf':
	import tensorflow as tf


class discriminator:

	def __init__(self, targets, n = 3, k = 2, target_probs = None,
						cycle_threshold = 0.7):

		self.n = n
		self.k = k
		self.wires = list(range(n))

		if target_probs is None:
			self.targets = [targets]
			self.target_probs = [1.]
		else:
			self.targets = targets
			self.target_probs = target_probs

		ops, pauli_strings, op_map = enumerate_pauli_qml_ops(self.n,self.k, True)
		self.operators = ops
		self.pauli_strings = pauli_strings
		self.qubit_operator_map = op_map

		self.operator_values = None
		self.active_operators = None
		self.active_coeff = None

		self.dev_dis = qml.device(	config.device_type, 
									wires=self.n, 
									analytic = config.analytic, 
									shots = config.shots)
		# self.dev_dis = qml.device(	"qiskit.aer", wires=n)


		self.cycle_threshold = cycle_threshold
		self.n_ops = len(ops)

		self.get_target_values()
		self.initialize_optimizer()


	def cycle_bad_operators(self, cycle_threshold = None, super_cycle = False):
		if cycle_threshold is not None:
			self.cycle_threshold = cycle_threshold
		op_mag = np.abs(self.terms.value)
		if super_cycle:
			op_mag = op_mag/np.sum(self.qubit_operator_map, axis = 0)
		active_op_term_min = np.min(op_mag[self.active_paulis])
		paulis_to_cycle = np.argwhere(op_mag < active_op_term_min*self.cycle_threshold).reshape(-1)
		paulis_to_keep = np.argwhere(op_mag >= active_op_term_min*self.cycle_threshold).reshape(-1)

		ops, pauli_strings, op_map = get_random_poly_ops(	n=self.n, 
															ignore_set = [self.pauli_strings[i] for i in paulis_to_keep],
															n_ops = len(paulis_to_cycle), 
															return_map = True) 
		
		# print('cycling:')
		# print(paulis_to_cycle)
		# print([self.pauli_strings[i] for i in self.active_paulis])
		# print(pauli_strings)
		for i,index_i in enumerate(paulis_to_cycle):
			self.operators[index_i] = ops[i]
			self.pauli_strings[index_i] = pauli_strings[i]
			self.qubit_operator_map[:,index_i] = op_map[:,i]
		# print(self.pauli_strings)

		self.get_target_values()
		self.reinitialize_optimizer()



	def get_target_values(self):


		def op_as_given_state(params, wires = self.wires):
			qml.QubitStateVector(params, wires = wires)

		if config.interface == 'torch':
			self.target_operator_values = torch.zeros(len(self.operators), device = config.torch_device)
		elif config.interface == 'tf':
			self.target_operator_values = tf.zeros([len(self.operators)], dtype = tf.double)
		# self.target_operator_values = np.zeros(len(self.operators))
		for i, target in enumerate(self.targets):
			qnodes = qml.map(op_as_given_state, self.operators, self.dev_dis, 
							measure="expval", interface = config.interface, diff_method=config.diff_method)
			if config.interface == 'torch':
				self.target_operator_values += self.target_probs[i]*qnodes( target ).to(device = config.torch_device)
			elif config.interface == 'tf':
				self.target_operator_values += self.target_probs[i]*tf.reshape(qnodes( target ), [-1])

		return self.target_operator_values


	def get_operator_values(self, gen, backend = None, n_evals = -1):
		self.operator_values = gen.measure_operators(self.operators)


	def initialize_optimizer(self):
		self.weights = cvxpy.Parameter(self.qubit_operator_map.shape, nonneg = True)
		self.weights.value = self.qubit_operator_map
		self.terms = cvxpy.Parameter((len(self.operators)))
		self.x = cvxpy.Variable((len(self.operators)))
		self.constraints = [self.weights@cvxpy.abs(self.x) <= (1./2.)]
		self.prob = cvxpy.Problem(cvxpy.Maximize(cvxpy.sum(self.terms*self.x)),self.constraints)

	def reinitialize_optimizer(self):
		self.weights.value = self.qubit_operator_map


	def find_optimal_operator(self, gen, zero_threshold = 1E-4, print_times = False, **kwargs):	

		if print_times:
			curr_time = time.time()
		self.get_operator_values(gen, **kwargs)
		if print_times:
			print('   took ' + str(time.time() - curr_time) + ' seconds to estimate Paulis')
			curr_time = time.time()

		if config.interface == 'torch':
			self.terms.value = self.operator_values.cpu().data.numpy() - self.target_operator_values.cpu().data.numpy()
		elif config.interface == 'tf':
			self.terms.value = self.operator_values.numpy().reshape(-1) - self.target_operator_values.numpy().reshape(-1)
		else:
			self.terms.value = self.operator_values - self.target_operator_values

		self.prob.solve(warm_start = True)

		if print_times:
			print('   took ' + str(time.time() - curr_time) + ' seconds to find active Paulis')
			curr_time = time.time()

		active_paulis = np.where(np.abs(self.x.value) > zero_threshold)[0].tolist()
		self.active_paulis = active_paulis
		active_operators = [self.operators[i] for i in active_paulis]
		active_coeff = [self.x.value[i] for i in active_paulis]
		active_terms =  [self.terms.value[i] for i in active_paulis]

		self.active_operators = active_operators
		if config.interface == 'torch':
			self.active_coeff = torch.tensor(active_coeff, device = config.torch_device)
			self.pauli_coeff = torch.tensor(self.x.value, device = config.torch_device)
			self.active_terms = torch.tensor(active_terms, device = config.torch_device)
		elif config.interface == 'tf':
			self.active_coeff = tf.reshape(tf.convert_to_tensor(active_coeff), [-1])
			self.pauli_coeff = tf.reshape(tf.convert_to_tensor(self.x.value), [-1])
			self.active_terms = tf.reshape(tf.convert_to_tensor(active_terms), [-1])
		else:
			self.active_coeff = active_coeff
			self.pauli_coeff = self.x.value
			self.active_terms = active_terms


	def get_target_expectation(self):
		if config.interface == 'torch':
			return torch.sum(self.pauli_coeff*self.target_operator_values)
		elif config.interface == 'tf':
			return tf.reduce_sum(self.pauli_coeff*self.target_operator_values)
		else:
			return np.sum(self.pauli_coeff*self.target_operator_values)


class generator:

	def __init__(self,circuits, params, n, param_values = None, probs = None):
		
		if probs is None:
			if config.interface == 'torch':
				self.probs = torch.tensor(np.ones(len(circuits))/len(circuits), requires_grad = True, device = config.torch_device)
			elif config.interface == 'tf':
				self.probs = tf.Variable(tf.ones((len(circuits)), dtype = tf.double)/len(circuits))
			else:
				self.probs = np.ones(len(circuits))/len(circuits)
		else:
			if config.interface == 'torch':
				self.probs = torch.tensor(probs, device = config.torch_device)
			elif config.interface == 'tf':
				self.probs = tf.Variable(tf.convert_to_tensor(np.asarray(probs), dtype = tf.double))
			else:
				self.probs = probs
		

		self.params = params
		self.n_circuits = len(circuits)
		self.n = n
		self.circuits = circuits

		self.dev_gen = [qml.device(	config.device_type, 
									wires=self.n, 
									analytic = config.analytic, 
									shots = config.shots) \
						for i in range(self.n_circuits)]
		# self.dev_gen = [qml.device(	"qiskit.aer", wires=n) \
		# 				for i in range(self.n_circuits)]


	def normalize_probs(self):
		# if config.interface == 'torch':
		# 	self.probs = relu(self.probs)
		# 	self.probs = self.probs / torch.sum(self.probs)
		# else:
		# 	self.probs = np.maximum(self.probs,0)
		# 	self.probs = self.probs/np.sum(self.probs)

		if config.interface == 'torch':
			self.probs_norm = softmax(self.probs, dim = 0)
		elif config.interface == 'tf':
			self.probs_norm = tf.exp(self.probs) / tf.reduce_sum(tf.exp(self.probs))

		

	def measure_hamiltonian(self, ops, coeffs, is_loss = True):

		output = 0
		self.normalize_probs()

		for i, circuit in enumerate(self.circuits):
			qnodes = qml.map(circuit, ops, self.dev_gen[i], 
							measure="expval", interface = config.interface, diff_method=config.diff_method)
			measurements = qnodes( self.params[i] )
			if config.interface == 'tf':
				measurements = tf.reshape(measurements, [-1])
			if config.interface == 'torch':
				output += self.probs_norm[i]*torch.sum(measurements*coeffs)
			elif config.interface == 'tf':
				output += self.probs_norm[i]*tf.reduce_sum(measurements*coeffs)

		if is_loss:
			self.loss = output

		return output

	def measure_operators(self,ops):
		if config.interface == 'torch':
			output = torch.zeros(len(ops), device = config.torch_device)
		elif config.interface == 'tf':
			output = tf.zeros([len(ops)], dtype = tf.double)

		self.normalize_probs()
		for i, circuit in enumerate(self.circuits):
			qnodes = qml.map(circuit, ops, self.dev_gen[i], 
							measure="expval", interface = config.interface, diff_method=config.diff_method)
			measurements = qnodes( self.params[i] )
			if config.interface == 'tf':
				measurements = tf.reshape(measurements, [-1])
			# print(measurements)
			# print(output)
			output += self.probs_norm[i]*measurements

		return output

	def get_state(self, circuit_num = 0):
		qnodes = qml.map(self.circuits[circuit_num], [qml.Identity(0)], self.dev_gen[circuit_num], 
						measure="expval", interface = config.interface, diff_method=config.diff_method)
		_ = qnodes( self.params[circuit_num] )
		return self.dev_gen[circuit_num]._state

	def get_all_states(self, return_as_density_mat = False):
		states = []
		for i in range(self.n_circuits):
			if config.interface == 'torch':
				states.append(self.get_state(i).reshape(-1))
			elif config.interface == 'tf':
				state = self.get_state(i)
				if isinstance(state, np.ndarray):
					states.append(state.reshape(-1))
				else:
					states.append(state.numpy().reshape(-1))

		if return_as_density_mat:
			if config.interface == 'torch':
				return states_to_density_mat(states, self.probs_norm.cpu().data.numpy())
			elif config.interface == 'tf':
				return states_to_density_mat(states, self.probs_norm.numpy())
		else:
			return states

class QGAN:
	def __init__(self, gen, dis):
		self.gen = gen
		self.dis = dis
		self.step = 0

	def get_optimizer(self, optimizer, **kwargs):
		if config.interface == 'torch':
			if optimizer == 'Adam':
				self.optimizer = torch.optim.Adam(self.gen.params+[self.gen.probs], **kwargs)
			elif optimizer == 'rms':
				self.optimizer = torch.optim.RMSprop(self.gen.params+[self.gen.probs], **kwargs)
			else:
				self.optimizer = torch.optim.SGD(self.gen.params+[self.gen.probs], **kwargs)
		elif config.interface == 'tf':
			if optimizer == 'Adam':
				self.optimizer = tf.keras.optimizers.Adam(**kwargs)
			elif optimizer == 'rms':
				self.optimizer = tf.keras.optimizers.RMSprop(**kwargs)
			else:
				self.optimizer = tf.keras.optimizers.SGD(**kwargs)

	def get_gen_grads(self, normalize = False):
		# setting up optimizer that takes a zero step
		if config.interface == 'torch':
			self.optimizer = torch.optim.SGD(self.gen.params, lr = 0.)
		# elif config.interface == 'tf':
			# self.optimizer = tf.keras.optimizers.SGD(learning_rate = 0.)
			# pass

		self.dis.find_optimal_operator(self.gen)
		if config.interface == 'torch':
				ham_loss = self.optimizer.step(self.closure)
				if normalize:
					ham_loss = ham_loss/self.gen.n
				gradients = self.gen.params.grad.cpu().numpy()
		elif config.interface == 'tf':
			with tf.GradientTape() as tape:
				ham_loss = self.gen.measure_hamiltonian(self.dis.active_operators, self.dis.active_coeff)
				if normalize:
					ham_loss = ham_loss/self.gen.n
			gradients = tape.gradient(ham_loss, self.gen.params)[0].numpy()

		return gradients

	def optimize(self, steps = 1000, verbose = False, print_times = False, optimizer = 'Adam', 
		cycle_ops = False, cycle_frequency = 10, cycle_threshold = 0.9, super_cycle = False, save_every = None,
		data_output_file = None, calc_pure_fidelity = False, calc_trace_distance = False, added_fields = None, 
		fidelity_stop = 1.01, **kwargs ):
		

		self.get_optimizer(optimizer, **kwargs)

		if data_output_file is not None and self.step == 0:
			self.setup_data_dict(calc_pure_fidelity = calc_pure_fidelity, calc_trace_distance = calc_trace_distance)

		if calc_trace_distance:
			dis_density_mat = states_to_density_mat(self.dis.targets, self.dis.target_probs)

		for step_i in range(steps):
			self.step = self.step + 1
			print('step '+str(self.step))

			if cycle_ops and step_i != 0 and (step_i % cycle_frequency) == 0:
				print('cycling operators')
				self.dis.cycle_bad_operators(cycle_threshold, super_cycle = super_cycle)

			if print_times:
				curr_time = time.time()
			self.dis.find_optimal_operator(self.gen,  print_times = print_times)
			if print_times:
				print('discriminator step took ' + str(time.time()-curr_time) + ' seconds')
				curr_time = time.time()

			if config.interface == 'torch':
				ham_loss = self.optimizer.step(self.closure)
			elif config.interface == 'tf':
				# try:
				with tf.GradientTape() as tape:
					ham_loss = self.gen.measure_hamiltonian(self.dis.active_operators, self.dis.active_coeff)

				gradients = tape.gradient(ham_loss, self.gen.params+[self.gen.probs])
				self.optimizer.apply_gradients(zip(gradients, self.gen.params+[self.gen.probs]))

				# except Exception as e:
				# 	print('gradient error: skipping step')
				# 	print(e)
				# 	print()
				# 	raise

			if print_times:
				print('generator step took ' + str(time.time()-curr_time) + ' seconds')
				curr_time = time.time()

			if config.interface == 'torch':
				est_W1_loss = (ham_loss-self.dis.get_target_expectation()).tolist()
			elif config.interface == 'tf':
				est_W1_loss = (ham_loss.numpy()-self.dis.get_target_expectation().numpy()).tolist()

			print('estimated W1 loss: ' + str(est_W1_loss))

			if calc_pure_fidelity:
				# state_out = self.gen.dev_gen[0]._state.reshape(-1)
				if config.interface == 'torch':
					state_out = self.gen.get_state().reshape(-1)
				elif config.interface == 'tf':
					state_out = self.gen.get_state()
					if isinstance(state_out, np.ndarray):
						state_out = state_out.reshape(-1)
					else:
						state_out = state_out.numpy().reshape(-1)
				pure_fid = pure_state_fidelity(state_out, self.dis.targets[0])
				print('pure state fidelity: ' + str(pure_fid))
				if pure_fid > fidelity_stop:
					break

			if calc_trace_distance:
				gen_density_mat = self.gen.get_all_states(return_as_density_mat = True)
				trace_dis = trace_distance(gen_density_mat, dis_density_mat)
				print('trace_distance: ' + str(trace_dis))

			if verbose:
				if config.interface == 'torch':
					print('circuit probabilities: ' + str(self.gen.probs_norm.tolist()))
				elif config.interface == 'tf':
					print('circuit probabilities: ' + str(self.gen.probs_norm.numpy().tolist()))
					print('operator values: ' + str( self.dis.active_terms.numpy().tolist() ))
				print('generator parameters: ')
				print(self.gen.params)
				print('active operators: ')
				print(self.dis.active_operators)
				# print(self.dis.active_coeff)
				# print(self.dis.active_terms)
				print()

			if data_output_file is not None:
				self.update_data_dict(	self.step, est_W1_loss, 
										pure_fidelity = pure_fid if calc_pure_fidelity else None,
										trace_dis = trace_dis if calc_trace_distance else None)

			if save_every is not None:
				if (step_i%save_every) == 0:
					self.save_data_dict(data_output_file, added_fields = added_fields)

		if data_output_file is not None:
			self.save_data_dict(data_output_file, added_fields = added_fields)


	def closure(self):
		self.optimizer.zero_grad()
		loss = self.gen.measure_hamiltonian(self.dis.active_operators, self.dis.active_coeff)
		loss.backward()
		return loss


	def setup_data_dict(self, calc_pure_fidelity = False, calc_trace_distance = False):
		self.data_dict = {	'step': [],
							'estimated_W1_loss': []}
		if calc_pure_fidelity:
			self.data_dict.update({'pure_state_fidelity': []})
		if calc_trace_distance:
			self.data_dict.update({'trace_distance': []})

	def update_data_dict(self, step, ham_loss, pure_fidelity = None, trace_dis = None):
		self.data_dict['step'] += [step]
		self.data_dict['estimated_W1_loss'] += [ham_loss]
		if pure_fidelity is not None:
			self.data_dict['pure_state_fidelity'] += [pure_fidelity]
		if trace_dis is not None:
			self.data_dict['trace_distance'] += [trace_dis]

	def save_data_dict(self,save_name, added_fields = None):
		self.data_dict.update({	'n_qubits': self.dis.n,
								'discriminator_pauli_order': self.dis.k,
								'n_generator_circuits': self.gen.n_circuits})

		if added_fields is not None:
			self.data_dict.update(added_fields)

		df = pd.DataFrame.from_dict(self.data_dict)
		df.to_csv('./csv_files/' + save_name)




class fidelity_qgan:
	def __init__(self, gen, dis):
		self.gen = gen
		self.dis = dis

	def get_gen_grads(self):
		# setting up optimizer that takes a zero step
		if config.interface == 'torch':
			self.optimizer = torch.optim.SGD(self.gen.params, lr = 0.)
		# elif config.interface == 'tf':
			# self.optimizer = tf.keras.optimizers.SGD(learning_rate = 0.)
			# pass

		self.dis.find_optimal_operator(self.gen)
		if config.interface == 'torch':
				raise ValueError('fidelity gradients not setup for pytorch configuration')
		elif config.interface == 'tf':
			with tf.GradientTape() as tape:
				state = tf.reshape(self.gen.get_state(), [-1])
				tar_state = tf.reshape(self.dis.targets[0], [-1])
				loss = 1 - tf.abs(tf.reduce_sum((tf.math.conj(state)*tar_state)))
			gradients = tape.gradient(loss, self.gen.params)[0].numpy()

		return gradients






if __name__ == '__main__':
	pass