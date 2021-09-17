# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from custom_circuits import butterfly_Pauli_constructor, butterfly_Pauli_constructor_simple
import general_utils
from QGAN import generator, discriminator, QGAN
import numpy as np
import config
import math

n = 4
k = 2

for gen_depth in [1,2,4]:
	for rank_target in [1,2,3]:
		for n_circuits in [rank_target, rank_target*2]:
			for target_depth in [1]:
				for simulation_i in range(5):

					save_name = 'simple_butterfly_cycled_n' + str(n) + '_k' + str(k) + \
								 '_ncirc' + str(n_circuits) + '_gend' + str(gen_depth) + \
								 '_tarrank' + str(rank_target) + '_tard' + str(target_depth) + \
								 '_simnum' + str(simulation_i) + '.csv'
					print(save_name)


					# for generating targets
					# circuit_data = [butterfly_Pauli_constructor(n, depth = target_depth, return_n_params = True) for i in range(rank_target)]
					circuit_data = [butterfly_Pauli_constructor_simple(n, depth = target_depth, return_n_params = True) for i in range(rank_target)]
					circuits = [circuit_data[i][0] for i in range(rank_target)]
					n_params_list = [circuit_data[i][1] for i in range(rank_target)]

					params = [general_utils.get_random_normal_params(n_param_i, format = config.interface) for n_param_i in n_params_list]

					gen = generator(circuits,params,n)
					target_states = gen.get_all_states()
					target_probs = np.random.rand(rank_target)
					target_probs = target_probs/np.sum(target_probs)

					del gen, params, circuit_data, circuits, n_params_list


					# constructing GAN
					circuit_data = [butterfly_Pauli_constructor_simple(n, depth = gen_depth, return_n_params = True) for i in range(n_circuits)]
					circuits = [circuit_data[i][0] for i in range(n_circuits)]
					n_params_list = [circuit_data[i][1] for i in range(n_circuits)]

					params = [general_utils.get_random_normal_params(n_param_i, format = config.interface) for n_param_i in n_params_list]

					dis = discriminator(target_states,n,k, target_probs = target_probs)
					gen = generator(circuits,params,n)
					qgan = QGAN(gen,dis)

					added_field_dict = {'generator_butterfly_depth': gen_depth,
										'target_density_matrix_rank': rank_target,
										'target_butterfly_depth': target_depth}
					qgan.optimize(	optimizer = 'Adam', learning_rate = 0.005, steps = 1000, cycle_ops = True, cycle_threshold = 0.8,
									calc_trace_distance = True, data_output_file = save_name, 
									added_fields = added_field_dict, save_every = 10, verbose = True)
