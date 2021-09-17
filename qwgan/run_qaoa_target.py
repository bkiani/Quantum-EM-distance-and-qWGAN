# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from custom_circuits import qaoa_translation_circuit, GHZ_constructor, butterfly_Pauli_constructor_simple
import general_utils
from QGAN import generator, discriminator, QGAN
import numpy as np
import math


def get_qaoa_translation_invariant_optimal_state(coeff, n):
	state = np.asarray(np.zeros(2**n).astype(np.complex128))
	if coeff < 0:
		if (n%2) == 0:
			optimal_1 = int('01'*int(n/2), 2)
			optimal_2 = int('10'*int(n/2), 2)
		else:
			optimal_1 = int('0' + '10'*int(n/2), 2)
			optimal_2 = int('1' + '01'*int(n/2), 2)
	else:
		optimal_1 = 0
		optimal_2 = -1

	state[optimal_1] += 1./np.sqrt(2)
	state[optimal_2] += 1./np.sqrt(2)
	return state


for n in [6,8]:
	for k in [2]:
		for depth in [2]:
			for trial_i in range(5):
				n_circuits = 1

				save_name = 'qaoa_translation_invariant_' + str(n) + '_k' + str(k) + \
							 '_ncirc' + str(n_circuits) + '_depth' + str(depth) + \
							 '_trial' + str(trial_i) + '.csv'
				print(save_name)

				circuit_data = [butterfly_Pauli_constructor_simple(n, True) for i in range(n_circuits)]
				# circuit_data = [qaoa_translation_circuit(n, True, depth = depth) for i in range(n_circuits)]
				circuits = [circuit_data[i][0] for i in range(n_circuits)]
				n_params_list = [circuit_data[i][1] for i in range(n_circuits)]

				params = [general_utils.get_random_normal_params(n_param_i, format = 'tf') for n_param_i in n_params_list]

				coeff = np.random.rand()-1.0
				state = get_qaoa_translation_invariant_optimal_state(coeff, n)

				dis = discriminator(state,n,k)
				gen = generator(circuits,params,n)
				qgan = QGAN(gen,dis)

				added_field_dict = {'qaoa_depth': depth}
				qgan.optimize(optimizer = 'Adam', learning_rate = 0.0035, steps = 1000, cycle_ops = True, cycle_threshold = 0.8,
								cycle_frequency = 5, calc_pure_fidelity = True, added_fields = added_field_dict,
								data_output_file = save_name, verbose = False, save_every = 10, print_times = False)
				