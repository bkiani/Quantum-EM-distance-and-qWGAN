import os
import glob
import pandas as pd
import numpy as np

default_csv_dir = '../qwgan/csv_files/'

def get_pandas_data(qualifier = '*.csv', n= None, csv_dir = None):
	if csv_dir is None:
		csv_dir = default_csv_dir

	if isinstance(n, int):
		n = [n]

	files_raw = glob.glob(csv_dir+qualifier)
	pd_files = [pd.read_csv(i) for i in files_raw]


	if n is not None:
		keep_nums = []
		for i,df in enumerate(pd_files):
			if df['n_qubits'][0] in n:
				keep_nums += [i]

		pd_files = [pd_files[i] for i in range(len(pd_files)) if i in keep_nums]

	return pd_files

def get_qaoa_data(	qualifier = '*qaoa_*', 
						n=4, k=2, 
						ncirc = 1, depth = 1,
						target_depth = None,
						csv_dir = None):
	if csv_dir is None:
		csv_dir = default_csv_dir

	files_raw = glob.glob(csv_dir+qualifier+'.csv')
	# print(files_raw)
	pd_files = [pd.read_csv(i) for i in files_raw]


	if n is not None:
		keep_nums = []
		for i,df in enumerate(pd_files):
			# print(df)
			if 	df['n_qubits'][0] == n and \
				df['discriminator_pauli_order'][0] == k and \
				df['n_generator_circuits'][0] == ncirc and \
				df['qaoa_depth'][0] == depth:
					if target_depth is not None:
						if df['depth_target'][0] == target_depth:
							keep_nums += [i]
					else:
						keep_nums += [i]

		pd_files = [pd_files[i] for i in range(len(pd_files)) if i in keep_nums]

	return pd_files


def summarize_gradient_data(filename, normalize_W1 = False, n_per_circuit = None):
	df = pd.read_csv(default_csv_dir + filename)
	groupby_fields = ['is_EM_QGAN','n_qubits']
	if n_per_circuit is None:
		n_per_circuit = np.max(df['n_trial'])

	if normalize_W1:
		# mask = df['is_EM_QGAN'] == True
		df['l1_norm'] =  df['l1_norm'] / df['n_qubits']
		df['l2_norm'] =  df['l2_norm'] / np.sqrt(df['n_qubits'])
		df['linf_norm'] =  df['linf_norm'] / np.log(df['n_qubits'])
		# df[df['is_EM_QGAN'] == True]['l1_norm'] =  df[df['is_EM_QGAN'] == True]['l1_norm'] / df[df['is_EM_QGAN'] == True]['n_qubits']
		# df[df['is_EM_QGAN'] == True]['l2_norm'] =  df[df['is_EM_QGAN'] == True]['l2_norm'] / np.sqrt(df[df['is_EM_QGAN'] == True]['n_qubits'])
		# df[df['is_EM_QGAN'] == True]['linf_norm'] =  df[df['is_EM_QGAN'] == True]['linf_norm'] / np.log(df[df['is_EM_QGAN'] == True]['n_qubits'])

	means = df.groupby(groupby_fields, group_keys = False).mean()
	variances = np.sqrt(df.groupby(groupby_fields, group_keys = False).var() / (df.groupby(groupby_fields, group_keys = False).count()/n_per_circuit))
	means.reset_index(level=means.index.names, inplace=True)
	variances.reset_index(level=variances.index.names, inplace = True)
	return means,variances



def get_percentiles(dfs, label, return_raw_data = True, percentiles = [10,25,50,75,90]):
	raw_data = np.zeros( (len(dfs), len(dfs[0][label])) )
	for i, df in enumerate(dfs):
		raw_data[i,:] = df[label]

	percentile_data = []
	for p in percentiles:
		percentile_data.append(np.percentile(raw_data,p,axis = 0))
	if return_raw_data:
		return percentile_data, raw_data
	else:
		return percentile_data

	
def get_butterfly_data(	qualifier = '*butterfly*', 
						n=4, k=2, 
						ncirc = 1, gend = 1, 
						tarrank = 1, tard = 1,
						csv_dir = None):
	if csv_dir is None:
		csv_dir = default_csv_dir

	files_raw = glob.glob(csv_dir+qualifier+'.csv')
	pd_files = [pd.read_csv(i) for i in files_raw]


	if n is not None:
		keep_nums = []
		for i,df in enumerate(pd_files):
			if 	df['n_qubits'][0] == n and \
				df['discriminator_pauli_order'][0] == k and \
				df['n_generator_circuits'][0] == ncirc and \
				df['generator_butterfly_depth'][0] == gend and \
				df['target_density_matrix_rank'][0] == tarrank and \
				df['target_butterfly_depth'][0] == tard:
				keep_nums += [i]

		pd_files = [pd_files[i] for i in range(len(pd_files)) if i in keep_nums]

	return pd_files


if __name__ == '__main__':
	m,v = summarize_gradient_data('gradients_depth2_bp.csv')
	print(v)