import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import plot_utils
import pandas as pd


figures_subfolder = 'figures/'

fig_width_small = 2.5
fig_width_med = 5.0
fig_width_onecolum = 3.5
fig_width_max = 7.3
fig_height_max = 6.0
fig_height_med = 3.0
fig_height_small = 2.0

default_plot_style = ['seaborn-whitegrid', './custom_style.mplstyle']
color_wheel = ['tab:gray', 'tab:blue', 'tab:orange']


def plot_success_rate(csv_name_fid, csv_name_wass, save_name):
	plt.style.use(default_plot_style)

	fig = plt.figure(constrained_layout=True)

	df_fid = pd.read_csv('../toy_model/csv_files/' + csv_name_fid)	
	df_fid['success'] = df_fid['success']*1
	out_data = df_fid.groupby("n").agg("mean")
	print(out_data)

	plt.plot(out_data.index, out_data['success'], marker = '.')

	df_wass = pd.read_csv('../toy_model/csv_files/' + csv_name_wass)	
	df_wass['success'] = df_wass['success']*1
	out_data = df_wass.groupby("n").agg("mean")
	print(out_data)

	plt.plot(out_data.index, out_data['success'], marker = '.')

	ax = plt.gca()
	# ax.set_xscale('log', basex=2)
	ax.set_xlabel('number of qubits ($n$)')
	ax.set_ylabel('success probability')

	ax.legend(['fidelity', 'EM distance'])
	
	fig.set_size_inches(fig_width_small, fig_height_small)
	plt.savefig(figures_subfolder + save_name, bbox_inches = 'tight')



if __name__ == '__main__':
	plot_success_rate('GHZ_fidelity_adam.csv', 'GHZ_wass_data.csv', 'GHZ_success_rate.pdf')