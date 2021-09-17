import torch
import math
import pandas as pd


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

	device = 'cpu'		# choose pytorch device

	n_parallel = 100 	# number of simulations performed in parallel
	n_steps = 100000	# number of steps of optimization before stopping

	for n in [4,8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48]:
		for trial_i in range(10):
			theta0 = torch.rand((n_parallel,n), requires_grad = True, device = device).double()*math.pi*2
			theta = torch.tensor(theta0, requires_grad = True, device = device)

			# optimizer = torch.optim.SGD([theta], lr=0.03)
			optimizer = torch.optim.Adam([theta], lr=0.02)

			for i in range(n_steps):
				optimizer.zero_grad()


				fidelity_each = 1-(torch.cos(theta[:,0])/math.sqrt(2) + torch.prod(torch.sin(theta), 1)/math.sqrt(2))**2
				fidelity = torch.sum(fidelity_each)
				fidelity.backward()
				optimizer.step()


			print(fidelity_each)
			fidelity_each = fidelity_each.cpu().data.numpy()
			first_qubit_val = torch.sin(theta[:,0]).cpu().data.numpy()
			success = fidelity_each < 0.02
			sim_ids = list(range(n_parallel))

			append_data()
			print(trial_i)
			print(i)
			print()

			save_to_csv('csv_files/GHZ_fidelity_adam.csv')
