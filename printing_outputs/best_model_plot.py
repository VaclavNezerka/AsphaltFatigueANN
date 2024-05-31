import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Enable LaTeX integration
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Defining options for datasets based on temperatures
twenty_deg = '20_deg'

# Defining options for loss optimization (either logarithmic or linear)
lin = 'lin'
log = 'log'

# Set actual options
actual_temp = twenty_deg
actual_opt = lin

# Load the Excel file
file_path = f'../excel_spreadsheets/{actual_opt}{actual_temp}_hyperparameter_tuning_results.xlsx'
df = pd.read_excel(file_path, usecols=['num_layers', 'dense', 'r2_test'])

# Define marker types based on unique values of 'dense' and 'num_layers'
dense_values = df['dense'].unique()
num_layer_values = df['num_layers'].unique()
marker_types = ['o', 's', '^', 'D', '*']  # Circle, Square, Triangle, Diamond, Star

# Map dense and num_layer values to marker types
dense_to_marker = {dense: marker for dense, marker in zip(dense_values, marker_types)}
num_layer_to_marker = {num_layer: marker for num_layer, marker in zip(num_layer_values, marker_types)}

# Define colors for different 'num_layers'
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

# Group by 'num_layers' and 'dense' and calculate mean and std of 'r2_test'
agg_df = df.groupby(['num_layers', 'dense'])['r2_test'].agg(['mean', 'std']).reset_index()

# Create a figure and a set of subplots
fig, axs = plt.subplots(1, 2, figsize=(10.5 * 0.75, 6 * 0.75))  # 1 row, 2 columns

# First subplot: mean r2_test vs. num_layers
ax = axs[0]
for _, row in agg_df.iterrows():
    num_layers = row['num_layers']
    dense = row['dense']
    mean = row['mean']
    ax.scatter(num_layers, mean, label=r'$n_{\mathrm{neur,h}}=%d$, $n_{\mathrm{h}}=%d$' % (dense, num_layers),
               color=colors[int(num_layers) % len(colors) - 1], marker=dense_to_marker[dense], facecolors='none')

ax.set_xlabel(r'$n_{\mathrm{h}}$')
ax.set_ylabel(r'$\overline{R^2}$')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_ylim(0, 1)

# Second subplot: mean r2_test vs. dense
ax = axs[1]
for _, row in agg_df.iterrows():
    num_layers = row['num_layers']
    dense = row['dense']
    mean = row['mean']
    ax.scatter(dense, mean, label=r'$n_{\mathrm{neur,h}}=%d$, $n_{\mathrm{h}}=%d$' % (dense, num_layers),
               color=colors[int(num_layers) % len(colors) - 1], marker=dense_to_marker[dense], facecolors='none')

ax.set_xlabel(r'$n_{\mathrm{neur,h}}$')
ax.set_ylabel(r'$\overline{R^2}$')
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_ylim(0, 1)

# Common legend
plt.subplots_adjust(top=0.85)
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.0))

# Adjust layout to fit the legend
fig.tight_layout(rect=[0, 0, 1, 0.75])

# Save the figure
output_dir = '../output/figures'
output_file_path = f'{output_dir}/{actual_opt}{actual_temp}_avg_r2_score.pdf'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
plt.savefig(output_file_path, format='pdf')
plt.show()