import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import plot_utils
import pandas as pd


figures_subfolder = 'figures/'


fig_width_med = 5.0
fig_width_onecolum = 3.5
fig_width_max = 7.3
fig_height_max = 6.0
fig_height_med = 3.0
fig_height_small = 2.0

default_plot_style = ['seaborn-whitegrid', './custom_style.mplstyle']
color_wheel = ['tab:gray', 'tab:blue', 'tab:orange']
percentile_linestyle = {10: 'dotted', #(0, (1, 10))
						25: 'dashed',
						50: 'solid',
						75: 'dashed',
						90: 'dotted'}


def plot_qaoa_losses(csv_identifier, save_name, n=[6,8], depths = [4], target_depth = None, width = fig_width_med):
	plt.style.use(default_plot_style)

	fig = plt.figure(constrained_layout=True)
	gs = fig.add_gridspec(	len(depths), len(n),
							wspace = 0.0, hspace = 0.05)

	a = []
	for i in range(len(n)):
		axes = []
		for j in range(len(depths)):
			axes.append(fig.add_subplot(gs[j,i]))
		a.append(axes)

	for i,n_i in enumerate(n):
		for j, depth_j in enumerate(depths):
			dfs = plot_utils.get_qaoa_data(csv_identifier, n=n_i, k=2, 
												ncirc = 1, depth = depth_j,
												target_depth = target_depth)
			plot_loss_in_axis(a[i][j],y_ids = ['estimated_W1_loss', 'pure_state_fidelity'], labels = ['est. EM loss', 'fidelity']) # setting up labels
			for df in dfs:
				plot_loss_in_axis(a[i][j], df, y_ids = ['estimated_W1_loss', 'pure_state_fidelity'], labels = ['est. EM loss', 'fidelity'], normalize_W1_loss = True)
			# a[i].set_title(str(n_i) + ' qubits')
			a[i][j].set_xlabel('optimization step')
			a[i][j].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc = 'lower left', ncol = 2)
			a[i][j].text(0.95, 0.12, '$n=' + str(n_i) + '$ \\ $L=' + str(depth_j) + '$',
						horizontalalignment='right',
					    verticalalignment='bottom',
						transform = a[i][j].transAxes)

	fig.set_size_inches(width, fig_height_small)
	plt.savefig(figures_subfolder + save_name, bbox_inches = 'tight')


def plot_bp_losses(csv_identifier, save_name, n=[6,8], depths = [4], target_depth = None, width = fig_width_med, limit = 5):
	plt.style.use(default_plot_style)

	fig = plt.figure(constrained_layout=True)
	gs = fig.add_gridspec(	len(depths), len(n),
							wspace = 0.0, hspace = 0.05)

	a = []
	for i in range(len(n)):
		axes = []
		for j in range(len(depths)):
			axes.append(fig.add_subplot(gs[j,i]))
		a.append(axes)

	for i,n_i in enumerate(n):
		for j, depth_j in enumerate(depths):
			dfs = plot_utils.get_qaoa_data(csv_identifier, n=n_i, k=2, 
												ncirc = 1, depth = depth_j,
												target_depth = target_depth)
			plot_loss_in_axis(a[i][j],y_ids = ['estimated_W1_loss', 'pure_state_fidelity'], labels = ['est. EM loss', 'fidelity']) # setting up labels
			for k, df in enumerate(dfs):
				if k >= limit:
					break
				plot_loss_in_axis(a[i][j], df, y_ids = ['estimated_W1_loss', 'pure_state_fidelity'], labels = ['est. EM loss', 'fidelity'], normalize_W1_loss = True)
			# a[i].set_title(str(n_i) + ' qubits')
			a[i][j].set_xlabel('optimization step')
			a[i][j].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc = 'lower left', ncol = 2)
			a[i][j].text(0.95, 0.12, '$n=' + str(n_i) + '$',
						horizontalalignment='right',
					    verticalalignment='bottom',
						transform = a[i][j].transAxes)

	fig.set_size_inches(width, fig_height_small)
	plt.savefig(figures_subfolder + save_name, bbox_inches = 'tight')



def plot_GHZ_losses(csv_identifier, n, save_name, limit = 5):
	plt.style.use(default_plot_style)

	fig = plt.figure(constrained_layout=True)
	gs = fig.add_gridspec(	1, len(n),
							wspace = 0.0, hspace = 0.05)

	a = []
	for i in range(len(n)):
		a.append(fig.add_subplot(gs[i]))

	for i,n_i in enumerate(n):
		dfs = plot_utils.get_pandas_data(csv_identifier, n_i)
		plot_loss_in_axis(a[i]) # setting up labels
		for k, df in enumerate(dfs):
			if k>= limit:
				break
			plot_loss_in_axis(a[i], df)
		# a[i].set_title(str(n_i) + ' qubits')
		a[i].set_xlabel('optimization step')
		a[i].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc = 'lower left', ncol = 2)
		a[i].text(0.05, 0.92, '$n=' + str(n_i) + '$')

	fig.set_size_inches(fig_width_max, fig_height_small)
	plt.savefig(figures_subfolder + save_name, bbox_inches = 'tight')


def plot_percentiles(csv_identifier, n, save_name, type = 'GHZ', **kwargs):
	plt.style.use(default_plot_style)

	fig = plt.figure(constrained_layout=True)
	if type == 'GHZ':
		dfs = plot_utils.get_pandas_data(csv_identifier, n)
	elif type == 'bp':
		dfs = plot_utils.get_qaoa_data(csv_identifier, n=n, **kwargs)

	ax = plt.gca()

	yids = ['estimated_W1_loss', 'pure_state_fidelity']
	percentiles = [10,25,50,75,90]
	p_labels = ['$10^{th}$ percentile', 
				'$25^{th}$ percentile',
				'median',
				'$75^{th}$ percentile',
				'$90^{th}$ percentile']


	w1_percentiles, w1_raw = plot_utils.get_percentiles(dfs, yids[0], percentiles = percentiles)
	fid_percentiles, fid_raw = plot_utils.get_percentiles(dfs, yids[1], percentiles = percentiles)


	ax.plot([],[],alpha=0.0, label = r'\underline{percentiles}')
	plot_percentiles_in_axis(ax, percentiles = percentiles, labels = p_labels)
	ax.plot([],[],alpha=0.0, label = r' ') # putting a blank space in the legend
	ax.plot([],[],alpha=0.0, label = r'\underline{metric}') 
	plot_loss_in_axis(ax)

	plot_percentiles_in_axis(ax,
							percentile_data = w1_percentiles,
							df_step = dfs[0]['step'],
							percentiles = percentiles,
							labels = None,
							normalize_factor = 1./dfs[0]['n_qubits'][0],
							color = color_wheel[0] )
	plot_percentiles_in_axis(ax,
							percentile_data = fid_percentiles,
							df_step = dfs[0]['step'],
							percentiles = percentiles,
							labels = None,
							normalize_factor = 1.,
							color = color_wheel[1] )
	for df in dfs:
		plot_loss_in_axis(ax, df, alpha = 0.15, linewidth = 0.5)

	ax.set_xlabel('optimization step')
	ax.set_ylabel('distance metric')
	ax.legend(bbox_to_anchor=(1.02, 1.02, 1., .102), loc = 'upper left')
	# ax.text(0.05, 0.92, '$n=' + str(n_i) + '$')

	fig.set_size_inches(fig_width_onecolum, fig_height_small)
	plt.savefig(figures_subfolder + save_name, bbox_inches = 'tight')




def plot_butterfly_losses(csv_identifier, save_name, n=4, gen_depth = 1, ranks = [1,2,3], ranks_gen = [1,2]):
	plt.style.use(default_plot_style)

	fig = plt.figure(constrained_layout=True)
	gs = fig.add_gridspec(	len(ranks_gen), len(ranks),
							wspace = 0.0, hspace = 0.05)

	a = []
	for i in range(len(ranks)):
		axes = []
		for j in range(len(ranks_gen)):
			axes.append(fig.add_subplot(gs[j,i]))
		a.append(axes)

	for i,rank_i in enumerate(ranks):
		for j, rank_j in enumerate(ranks_gen):
			rank_gen = int(rank_j*rank_i)
			dfs = plot_utils.get_butterfly_data(csv_identifier, n=n, k=2, 
												ncirc = rank_gen, gend = gen_depth, 
												tarrank = rank_i, tard = 1)
			plot_loss_in_axis(a[i][j],y_ids = ['estimated_W1_loss', 'trace_distance'], labels = ['est. EM loss', 'trace distance']) # setting up labels
			for df in dfs:
				plot_loss_in_axis(a[i][j], df, y_ids = ['estimated_W1_loss', 'trace_distance'], labels = ['est. EM loss', 'trace distance'], normalize_W1_loss = False)
			# a[i].set_title(str(n_i) + ' qubits')
			a[i][j].set_xlabel('optimization step')
			a[i][j].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc = 'lower left', ncol = 2)
			a[i][j].text(0.95, 0.95, '$r_{tar}=' + str(rank_i) + '$ \\ $r_{gen}=' + str(rank_gen) + '$',
						horizontalalignment='right',
					    verticalalignment='top',
						transform = a[i][j].transAxes)

	fig.set_size_inches(fig_width_max, fig_height_med)
	plt.savefig(figures_subfolder + save_name, bbox_inches = 'tight')

def plot_loss_in_axis(	ax, 
						df = None, 
						y_ids = ['estimated_W1_loss', 'pure_state_fidelity'],
						labels = ['est. EM loss', 'fidelity'],
						normalize_W1_loss = True,
						alpha = 0.9,
						linewidth = None):
	if linewidth is None:
		linewidth = matplotlib.rcParams['lines.linewidth']

	if df is None:
		for i, label in enumerate(labels):
			ax.plot([],[], color = color_wheel[i], label = label, alpha = alpha)
	else:
		for i, y_id in enumerate(y_ids):
			if y_id == 'estimated_W1_loss' and normalize_W1_loss:
				ax.plot(df['step'], df[y_id]/df['n_qubits'], color = color_wheel[i], alpha = alpha, linewidth = linewidth)
			else:
				ax.plot(df['step'], df[y_id], color = color_wheel[i], alpha = alpha, linewidth = linewidth)



def plot_percentiles_in_axis( 	ax,
								percentile_data = None,
								df_step = None,
								percentiles = [10,25,50,75,90],
								labels = [	'$10^{th}$ percentile', 
											'$25^{th}$ percentile',
											'median',
											'$75^{th}$ percentile',
											'$90^{th}$ percentile'],
								normalize_factor = 1.,
								color = 'black' ):

	if percentile_data is None:
		for i,label in enumerate(labels):
			ax.plot([],[],color = color, linestyle = percentile_linestyle[percentiles[i]], label = label)
	else:
		for i,perc in enumerate(percentiles):
			ax.plot(df_step,percentile_data[i]*normalize_factor,color = color, linestyle = percentile_linestyle[perc])

if __name__ == '__main__':

	plot_percentiles('*GHZ_simplecycled*.csv', 8, 'GHZ_cycled_8_percentiles.pdf', type = 'GHZ')
	plot_percentiles('*bp_depth4_nocycle_teach_stu_*', 8, 'bp_depth4_nocycle_teach_stu_8_percentiles.pdf', type = 'bp', depth = 4, target_depth = 2, k = 2, ncirc = 1)

	plot_GHZ_losses('*GHZ_simplecycled*.csv', [4,8,12], 'GHZ_cycled_4_8_12.pdf')
	plot_bp_losses('*bp_depth4_nocycle_teach_stu_*', 'bp_depth4_nocycle_teach_stu_4_6_8.pdf', n=[4,6,8], depths = [4], target_depth = 2, width = fig_width_max, limit=5)

	plot_butterfly_losses('*butterfly_cycled*', 'butterfly_cycled_4.pdf')
	plot_qaoa_losses('*qaoa_translation_invariant*', 'qaoa_translation_invariant_6_8.pdf')

