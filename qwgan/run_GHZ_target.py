# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

from custom_circuits import GHZ_constructor, GHZ_constructor_simple
import general_utils
from QGAN import generator, discriminator, QGAN
import numpy as np
import math
import config
# import torch


for n in [4,8,12]:
	for k in [2]:
		for trial_i in range(50):
			n_circuits = 1

			save_name = 'GHZ_simplecycled_n' + str(n) + '_k' + str(k) + \
						 '_ncirc' + str(n_circuits) + '_trial' + str(trial_i) + '.csv'
			print(save_name)

			circuit_data = [GHZ_constructor_simple(n, True) for i in range(n_circuits)]
			circuits = [circuit_data[i][0] for i in range(n_circuits)]
			n_params_list = [circuit_data[i][1] for i in range(n_circuits)]

			params = [general_utils.get_random_normal_params(n_param_i, format = config.interface) for n_param_i in n_params_list]

			state = np.asarray(np.zeros(2**n).astype(np.complex128))
			state[0] = np.sqrt(2)/2
			state[-1] = np.sqrt(2)/2

			dis = discriminator(state,n,k)
			gen = generator(circuits,params,n)
			qgan = QGAN(gen,dis)



			qgan.optimize(optimizer = 'Adam', learning_rate = 0.01, steps = 1000, cycle_ops = True, cycle_threshold = 0.8,
							calc_pure_fidelity = True, data_output_file = save_name, verbose = False,save_every = 10)
			