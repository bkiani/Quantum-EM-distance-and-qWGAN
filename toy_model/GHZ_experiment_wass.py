import torch
from torch.autograd import Variable
import math
import pandas as pd
import numpy as np
# from scipy.linalg import solve


def pauli_z(n,i):
	mat = np.eye(n+1)
	if i==0:
		mat = mat*-1
		mat[0,0] = 1
	else:
		mat[i+1:,i+1:] = -1*mat[i+1:,i+1:]
	return mat

# def multi_pauli_y(n,i):
# 	mat = np.zeros((n+1,n+1))
# 	mat[i+1,0] = -1.
# 	mat[0,i+1] = 1.
# 	return mat

def multi_pauli_x(n,i):
	mat = np.zeros((n+1,n+1))
	mat[i+1,0] = 1.
	mat[0,i+1] = 1.
	return mat


def get_paulis(n):
	pauli_list = np.zeros((2*n,n+1,n+1))
	for i in range(n):
		pauli_list[i,:,:] = pauli_z(n,i)
	for i in range(n):
		pauli_list[n+i,:,:] = multi_pauli_x(n,i)
	# for i in range(n):
	# 	pauli_list[2*n+i,:,:] = multi_pauli_x(n,i)
	return torch.tensor(pauli_list, requires_grad = True, device = device, dtype = dtype)


save_dict = {	'n': [],
				'trial_num': [],
				'experiment_id': [],
				'final_fidelity': [],
				'success': [],
				'step':	[],
				'first_qubit_val': []
			}

def save_to_csv(name):
	df = pd.DataFrame.from_dict(save_dict)
	df.to_csv(name)

def append_data():
	save_dict['n'] += [n for i in range(n_parallel)]
	save_dict['trial_num'] += [trial_i for i in range(n_parallel)]
	save_dict['experiment_id'] += [i for i in range(n_parallel)]
	save_dict['final_fidelity'] += fidelity_each.tolist()
	save_dict['success'] += success.tolist()
	save_dict['step'] += [n_steps for i in range(n_parallel)]
	save_dict['first_qubit_val'] += first_qubit_val.tolist()



if __name__ == '__main__':
	n_steps = 10000		# number of steps of optimization before stopping
	n_parallel = 100	# number of simulations performed in parallel
	device = 'cpu'		# choose pytorch device

	dtype = torch.float64

	for n in [4,8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48]:
		print(n)
		for trial_i in range(10):
			print('new trial')
			print(trial_i)
			print()

			target = np.zeros((n+1,1))
			target[0,0] = 1/np.sqrt(2)
			target[-1,0] = 1/np.sqrt(2)
			target = torch.tensor(target, requires_grad = True, device = device, dtype = dtype)

			paulis = get_paulis(n)

			target_expectations = torch.transpose(target,0,1)@paulis@target
			target_expectations = target_expectations.reshape(1,-1)
			# print(target_expectations)

			sin_filter = np.tril(np.ones((n+1,n)),-1)
			cos_filter = np.zeros((n+1,n))
			cos_filter[:-1,:] = np.eye(n)
			one_filter = np.triu(np.ones((1,n+1,n)),1)
			# sin_filter = torch.from_numpy(sin_filter, requires_grad = True, device = device, dtype = dtype)
			# cos_filter = torch.from_numpy(cos_filter, requires_grad = True, device = device, dtype = dtype)
			with torch.no_grad():
				sin_filter = torch.from_numpy(sin_filter.reshape(1,n+1,n)).type(dtype).requires_grad_(True).to(device)
				cos_filter = torch.from_numpy(cos_filter.reshape(1,n+1,n)).type(dtype).requires_grad_(True).to(device)
				one_filter = torch.from_numpy(one_filter.reshape(1,n+1,n)).type(dtype).requires_grad_(True).to(device)



			theta0 = torch.rand((n_parallel, 1, n), requires_grad = True, device = device, dtype = dtype)*math.pi*2
			# theta0 = np.random.rand(n_parallel, 1, n)*math.pi*2
			# theta = torch.tensor(theta0, requires_grad = True, device = device, dtype = dtype)
			theta = theta0.detach().clone().requires_grad_(True)

			# theta.expand(n_parallel, n+1, n)
			# print(theta)
			# theta_mat = torch.ones((n_parallel, n+1, n), requires_grad = True, device = device, dtype = dtype)
			# theta_mat = torch.triu(theta_mat, 1) + torch.triu(torch.cos(theta),0) - torch.triu(torch.cos(theta),1) + torch.tril(torch.sin(theta),-1)
			# print(theta_mat)
			# theta = torch.ones((n_parallel,1,n), requires_grad = True, device = device, dtype = dtype)
			# print(theta)
			# theta_mat = Variable(torch.mul(sin_filter,torch.sin(theta)) + torch.mul(cos_filter,torch.cos(theta))+ one_filter)


			# optimizer = torch.optim.SGD([theta], lr=0.2)
			optimizer = torch.optim.Adam([theta], lr=0.02)

			for i in range(n_steps):
				optimizer.zero_grad()
				# print()
				# print(i)

				# theta_mat = torch.mul(sin_filter,torch.sin(theta.clone())) + torch.mul(cos_filter,torch.cos(theta.clone()))+ one_filter
				theta_mat = torch.mul(sin_filter,torch.sin(theta)) + torch.mul(cos_filter,torch.cos(theta))+ one_filter
				state_local = torch.prod(theta_mat,-1)
				state = state_local.reshape(n_parallel, 1, -1, 1)

				# if i%10==0:
				# 	print(state)

				expectations = torch.abs((torch.transpose(state,-1,-2)@paulis.reshape(1,-1,n+1,n+1)@state).reshape(n_parallel,-1) - target_expectations)
				loss_temp, _ = torch.max(expectations, -1)
				# print(loss_temp)
				loss = torch.max(loss_temp, torch.sum(expectations[:,:n], -1))
				fidelity_each = ((torch.transpose(state,-1,-2)@target)**2).reshape(-1)
				if (i%1000) == 0:
					print()
					print(i)
					print(loss.reshape(-1))
					print(fidelity_each)
				loss = torch.sum(loss)
				loss.backward(retain_graph = True)
				# loss.backward()

				optimizer.step()

			fidelity_each = fidelity_each.cpu().data.numpy()
			first_qubit_val = torch.sin(theta[:,:,0]).cpu().data.numpy().reshape(-1)
			success = fidelity_each > 0.98
			sim_ids = list(range(n_parallel))
			append_data()
			del loss, theta, theta_mat, expectations,theta0,cos_filter,sin_filter,one_filter

			save_to_csv('csv_files/GHZ_wass_data.csv')
