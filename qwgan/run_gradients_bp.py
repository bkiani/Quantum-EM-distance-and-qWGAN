# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from custom_circuits import bp_circuit
import general_utils
from QGAN import generator, discriminator, QGAN, fidelity_qgan
import numpy as np
import math
import config
import pandas as pd


def initialize_pandas_dict():
	save_dict = {
					'n_qubits': [],
					'is_EM_QGAN': [],
					'discriminator_pauli_order': [],
					'n_generator_circuits': [],
					'depth_target': [],
					'depth': [],
					'GAN_num': [],
					'n_trial': [],
					'l1_norm': [],
					'l2_norm': [],
					'linf_norm': [],
					'first_grad_entry': [],
					'last_grad_entry': []
				}
	return save_dict

def update_pandas_dict(save_dict, is_EM_QGAN):
	save_dict['n_qubits'].append(n),
	save_dict['is_EM_QGAN'].append(is_EM_QGAN)
	save_dict['discriminator_pauli_order'].append(k)
	save_dict['n_generator_circuits'].append(n_circuits)
	save_dict['depth_target'].append(depth_target)
	save_dict['depth'].append(depth)
	save_dict['GAN_num'].append(network_i)
	save_dict['n_trial'].append(trial_i)
	save_dict['l1_norm'].append(np.sum(np.abs(grads)))
	save_dict['l2_norm'].append(np.sqrt(np.sum(np.abs(grads*grads))))
	save_dict['linf_norm'].append(np.max(np.abs(grads)))
	save_dict['first_grad_entry'].append(np.abs(grads.reshape(-1)[0]))
	save_dict['last_grad_entry'].append(np.abs(grads.reshape(-1)[-1]))
	return save_dict

def save_data():
	print('saving')
	df = pd.DataFrame.from_dict(save_dict)
	df.to_csv('./csv_files/' + save_name)	





# name of csv file where data is saved
save_name = 'gradients_depth2_bp_100.csv'


n_per_trial = 4
n_circuits = 1
save_dict = initialize_pandas_dict()

for n in [3,4,5,6,7,8,9,10,11,12,13,14]:
	for k in [2]:
		for depth_target in [2]:
			for depth in [2]:
				for network_i in range(25):
					print()
					print()
					print('N=' + str(n))
					print('network ' + str(network_i))


					# for generating targets
					circuit_data = [bp_circuit(n, return_n_params = True, depth = depth_target) for i in range(n_circuits)]
					circuits = [circuit_data[i][0] for i in range(n_circuits)]
					n_params_list = [circuit_data[i][1] for i in range(n_circuits)]

					params = [general_utils.get_random_normal_params(n_param_i, format = config.interface) for n_param_i in n_params_list]

					gen = generator(circuits,params,n)
					target_states = gen.get_all_states()
					target_probs = np.random.rand(n_circuits)
					target_probs = target_probs/np.sum(target_probs)

					del gen, params, circuits, n_params_list

					circuit_data = [bp_circuit(n, return_n_params = True, depth = depth) for i in range(n_circuits)]
					circuits = [circuit_data[i][0] for i in range(n_circuits)]
					n_params_list = [circuit_data[i][1] for i in range(n_circuits)]
					dis = discriminator(target_states,n,k, target_probs = target_probs)

					for trial_i in range(n_per_trial):
						print('trial ' + str(trial_i))
						params = [general_utils.get_random_normal_params(n_param_i, format = config.interface) for n_param_i in n_params_list]
						gen = generator(circuits,params,n)
						qgan = QGAN(gen,dis)
						grads = qgan.get_gen_grads()
						save_dict = update_pandas_dict(save_dict, True)

						fid_qgan = fidelity_qgan(gen,dis)
						grads = fid_qgan.get_gen_grads()
						save_dict = update_pandas_dict(save_dict, False)


					save_data()