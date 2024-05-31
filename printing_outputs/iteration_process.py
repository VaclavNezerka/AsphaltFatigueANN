import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

# Function to format numbers in scientific notation
def sci_notation(number, sig_fig=2):
    ret_string = "{0:.{1}e}".format(number, sig_fig)
    a, b = ret_string.split("e")
    b = int(b)
    return r'%s\times10^{%d}' % (a, b)

# Load the epoch logs from the Excel files
lin_epoch_log_path = pd.read_excel('../output/lin/20_deg/2layers_123dense_3175epochs_adam_optimizer_relu_activation/epoch_logs.xlsx')
log_epoch_log_path = pd.read_excel('../output/log/20_deg/1layers_10dense_2001epochs_adam_optimizer_relu_activation/epoch_logs.xlsx')
output_file_path = (f'../output/figures/iteration_process.pdf')

# Enable LaTeX integration
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Extracting data from DataFrames
num_epoch_lin = lin_epoch_log_path.iloc[:, 0].to_numpy()  # First column to array
train_loss_lin = lin_epoch_log_path.iloc[:, 1].to_numpy()  # Second column to array
val_loss_lin = lin_epoch_log_path.iloc[:, 2].to_numpy()  # Third column to array

num_epoch_log = log_epoch_log_path.iloc[:, 0].to_numpy()  # First column to array
train_loss_log = log_epoch_log_path.iloc[:, 1].to_numpy()  # Second column to array
val_loss_log = log_epoch_log_path.iloc[:, 2].to_numpy()  # Third column to array

# Adjust start index to skip the initial training phase (10%)
start_index_lin = int(len(num_epoch_lin) * 0.1)
start_index_log = int(len(num_epoch_log) * 0.1)

# Adjusted data
num_epoch_lin_adj = num_epoch_lin[start_index_lin:]
train_loss_lin_adj = train_loss_lin[start_index_lin:]
val_loss_lin_adj = val_loss_lin[start_index_lin:]

num_epoch_log_adj = num_epoch_log[start_index_log:]
train_loss_log_adj = train_loss_log[start_index_log:]
val_loss_log_adj = val_loss_log[start_index_log:]

# Find the minimum validation loss and corresponding epoch for both linear and logarithmic models
min_val_loss_lin = np.min(val_loss_lin_adj)
min_epoch_lin = num_epoch_lin_adj[np.argmin(val_loss_lin_adj)]
min_val_loss_log = np.min(val_loss_log_adj)
min_epoch_log = num_epoch_log_adj[np.argmin(val_loss_log_adj)]

# Create a figure and a set of subplots
fig, axs = plt.subplots(1, 2, figsize=(10.5 * 0.75, 5 * 0.75))  # Adjust size as needed

# Common plot settings
for ax in axs:
    ax.set_xlabel(r'Epoch ($\times 10^3$)')
    ax.set_ylabel(r'$L(y, \hat{y})$')

# Plotting for linear variant
axs[0].plot(num_epoch_lin_adj / 1000, train_loss_lin_adj, color='blue', label='Training', alpha=0.7)
axs[0].plot(num_epoch_lin_adj / 1000, val_loss_lin_adj, color='orange', label='Validation', alpha=0.7)
axs[0].axvline(x=min_epoch_lin / 1000, color='black', linestyle='--', alpha=0.7)
axs[0].scatter(min_epoch_lin / 1000, min_val_loss_lin, color='red', s=25, zorder=5)
axs[0].set_ylabel(r'$L_{\mathrm{MSE}}(y, \hat{y})$')

# Plotting for logarithmic variant
axs[1].plot(num_epoch_log_adj / 1000, train_loss_log_adj, color='blue', label='Training', alpha=0.7)
axs[1].plot(num_epoch_log_adj / 1000, val_loss_log_adj, color='orange', label='Validation', alpha=0.7)
axs[1].axvline(x=min_epoch_log / 1000, color='black', linestyle='--', alpha=0.7)
axs[1].scatter(min_epoch_log / 1000, min_val_loss_log, color='red', s=25, zorder=5)
axs[1].set_ylabel(r'$L_{\mathrm{MSLE}}(y, \hat{y})$')

# Create custom legend entries for minimum validation loss points
lin_1 = fr'min $L_{{\mathrm{{MSE}}}}(y, \hat{{y}}) = %s$' % sci_notation(min_val_loss_lin, sig_fig=2)
log_1 = fr'min $L_{{\mathrm{{MSLE}}}}(y, \hat{{y}}) = %s$' % sci_notation(min_val_loss_log, sig_fig=2)

# Custom legend markers
min_val_loss_marker_lin = Line2D([0], [0], marker='o', color='w', label=f'{lin_1}', markersize=5, markerfacecolor='red')
min_val_loss_marker_log = Line2D([0], [0], marker='o', color='w', label=f'{log_1}', markersize=5, markerfacecolor='red')

# Adding individual legends with custom legend entries for the min validation loss
axs[0].legend(handles=[min_val_loss_marker_lin], loc='upper center', fontsize=10)
axs[1].legend(handles=[min_val_loss_marker_log], loc='upper center', fontsize=10)

# Shared Legend for Training and Validation only
handles, labels = axs[0].get_legend_handles_labels()
unique_labels, unique_handles = [], []
for handle, label in zip(handles, labels):
    if label not in unique_labels:
        unique_labels.append(label)
        unique_handles.append(handle)

# Place the shared legend above the plots
fig.legend(unique_handles, unique_labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.0))

# Adjust layout
fig.tight_layout(rect=[0, 0.03, 1, 0.95])

# Save the figure
plt.savefig(output_file_path, format='pdf')
plt.show()