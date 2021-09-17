import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import plot_utils
import pandas as pd


figures_subfolder = 'figures/'


fig_width_med = 5.0
fig_width_onecolum = 3.5
fig_width_smallest = 2.7
fig_width_max = 7.3
fig_height_max = 6.0
fig_height_med = 3.0
fig_height_small = 2.0

default_plot_style = ['seaborn-whitegrid', './custom_style.mplstyle']
color_wheel = ['tab:gray', 'tab:blue', 'tab:orange']


def plot_gradients_in_axis(ax,df_data, df_error, dataid):
	QGAN = df_data[df_data['is_EM_QGAN'] == True]
	fidGAN = df_data[df_data['is_EM_QGAN'] == False]
	QGAN_err = df_error[df_error['is_EM_QGAN'] == True]
	fidGAN_err = df_error[df_error['is_EM_QGAN'] == False]
	# plt.sca(ax)
	ax.errorbar(QGAN['n_qubits'], QGAN[dataid], yerr= QGAN_err[dataid], marker = '.', markersize = 4)
	ax.errorbar(fidGAN['n_qubits'], fidGAN[dataid], yerr= 1.96*fidGAN_err[dataid], marker = '.', markersize = 4)

	ax.legend(['qWGAN metric', 'inner product metric'])
	ax.set_xlabel('num. qubits ($n$)')
	ax.set_ylabel('gradient value')
	ax.set_yscale('log')
	ax.set_xlim([3,15])



def plot_gradients(csv_identifier, save_name, normalize_W1 = True):
	plt.style.use(default_plot_style)

	fig = plt.figure(constrained_layout=True)
	gs = fig.add_gridspec(	1, 3,
							wspace = 0.0, hspace = 0.2)
	a = []
	for i in range(3):
		a.append(fig.add_subplot(gs[i]))
	df_data, df_error = plot_utils.summarize_gradient_data(csv_identifier, normalize_W1)

	dataids = ['l1_norm', 'l2_norm', 'first_grad_entry']
	texts = ['$\ell^l$ norm', '$\ell^2$ norm', 'grad. of first param.']
	for i, dataid  in enumerate(dataids):
		ax_i = a[i]
		plot_gradients_in_axis(ax_i, df_data, df_error, dataid)
		ax_i.set_title(texts[i])

	fig.set_size_inches(fig_width_max, fig_height_small)
	plt.savefig(figures_subfolder + save_name, bbox_inches = 'tight')


def plot_single_gradient(csv_identifier, save_name, dataid = 'l1_norm', ylabel = '$\ell^1$ norm of gradient vector', normalize_W1 = True):
	plt.style.use(default_plot_style)

	fig = plt.figure(constrained_layout=True)
	gs = fig.add_gridspec(	1, 1,
							wspace = 0.0, hspace = 0.2)
	a = []
	a.append(fig.add_subplot(gs[0]))
	df_data, df_error = plot_utils.summarize_gradient_data(csv_identifier, normalize_W1)

	ax_i = a[0]
	plot_gradients_in_axis(ax_i, df_data, df_error, dataid)
	ax_i.set_ylabel(ylabel)


	fig.set_size_inches(fig_width_smallest, fig_height_small)
	plt.savefig(figures_subfolder + save_name, bbox_inches = 'tight')


if __name__ == '__main__':
	plot_gradients('gradients_depth2_bp_100.csv', 'gradients_depth2_bp.pdf')
	plot_single_gradient('gradients_depth2_bp_100.csv', 'gradients_depth2_bp_l1.pdf')
