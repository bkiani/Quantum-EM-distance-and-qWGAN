# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

from custom_circuits import bp_circuit
import general_utils
from QGAN import generator, discriminator, QGAN
import numpy as np
import math
import config


for trial_i in range(50): # choose number of trials to perform
	for n in [4,6,8]: # change n to get different qubit numbers
		for k in [2]:
			for depth_target in [2]:
				for depth in [4]:
					n_circuits = 1

					save_name = 'bp_depth4_nocycle_teach_stu_' + str(n) + '_k' + str(k) + \
								 '_ncirc' + str(n_circuits) + '_depth' + str(depth) + \
								 '_depthtar' + str(depth_target) + \
								 '_trial' + str(trial_i) + '.csv'
					print(save_name)

					# for generating targets
					circuit_data = [bp_circuit(n, return_n_params = True, depth = depth_target) for i in range(n_circuits)]
					circuits = [circuit_data[i][0] for i in range(n_circuits)]
					n_params_list = [circuit_data[i][1] for i in range(n_circuits)]

					params = [general_utils.get_random_normal_params(n_param_i, format = config.interface) for n_param_i in n_params_list]

					gen = generator(circuits,params,n)
					target_states = gen.get_all_states()
					target_probs = np.random.rand(n_circuits)
					target_probs = target_probs/np.sum(target_probs)

					print('target parameters: ')
					print(params)

					del gen, params, circuits, n_params_list



					circuit_data = [bp_circuit(n, return_n_params = True, depth = depth) for i in range(n_circuits)]
					circuits = [circuit_data[i][0] for i in range(n_circuits)]
					n_params_list = [circuit_data[i][1] for i in range(n_circuits)]

					params = [general_utils.get_random_normal_params(n_param_i, format = config.interface) for n_param_i in n_params_list]

					dis = discriminator(target_states,n,k, target_probs = target_probs)
					gen = generator(circuits,params,n)
					qgan = QGAN(gen,dis)

					cycle = False
					super_cycle = False

					if config.interface == 'torch':
						optimizer_args = {	'lr': 0.02,
											'weight_decay': 0.00
											}
					else:
						optimizer_args = {	'learning_rate': 0.025
											}

					added_field_dict = {'qaoa_depth': depth,
										'depth_target': depth_target}
					qgan.optimize(optimizer = 'Adam', steps = 500, cycle_ops = cycle, cycle_threshold = 0.7, super_cycle = super_cycle,
									cycle_frequency = 5, calc_pure_fidelity = True, added_fields = added_field_dict,
									data_output_file = save_name, verbose = True, save_every = 10, print_times = False,
									fidelity_stop = 1.01,
									**optimizer_args)

					if config.interface == 'torch':
						optimizer_args = {	'lr': 0.007,
											'weight_decay': 0.00
											}
					else:
						optimizer_args = {	'learning_rate': 0.007
											}

					qgan.optimize(optimizer = 'Adam', steps = 500, cycle_ops = cycle, cycle_threshold = 0.7, super_cycle = super_cycle,
									cycle_frequency = 5, calc_pure_fidelity = True, added_fields = added_field_dict,
									data_output_file = save_name, verbose = True, save_every = 10, print_times = False,
									fidelity_stop = 1.01,
									**optimizer_args)
					