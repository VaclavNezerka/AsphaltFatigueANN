import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import os


# Function to format numbers in scientific notation
def sci_notation(number, sig_fig=2):
    """
    Converts a number to scientific notation with specified significant figures.

    Args:
    - number: The number to format.
    - sig_fig: Number of significant figures (default is 2).

    Returns:
    - A string formatted in LaTeX-style scientific notation.
    """
    ret_string = "{0:.{1}e}".format(number, sig_fig)
    a, b = ret_string.split("e")
    b = int(b)
    return r'%s\times10^{%d}' % (a, b)

# Enable LaTeX integration for better plot rendering
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Load the epoch logs from the Excel files for linear and logarithmic models
lin_epoch_log_path = pd.read_excel(
    '../output/lin/20_deg/1kf_2layers_200dense_28epochs_adam_optimizer_relu_activation/epoch_logs.xlsx')
log_epoch_log_path = pd.read_excel(
    '../output/log/20_deg/1kf_2layers_200dense_18epochs_adam_optimizer_relu_activation/epoch_logs.xlsx')

# Extract epoch and loss data from the DataFrames
# Linear model data
num_epoch_lin = lin_epoch_log_path.iloc[:, 0].to_numpy()  # Epochs
train_loss_lin = lin_epoch_log_path.iloc[:, 1].to_numpy()  # Training loss
val_loss_lin = lin_epoch_log_path.iloc[:, 2].to_numpy()  # Validation loss

# Logarithmic model data
num_epoch_log = log_epoch_log_path.iloc[:, 0].to_numpy()  # Epochs
train_loss_log = log_epoch_log_path.iloc[:, 1].to_numpy()  # Training loss
val_loss_log = log_epoch_log_path.iloc[:, 2].to_numpy()  # Validation loss

# Skip the initial training phase (10% of the epochs) for both models
start_index_lin = int(len(num_epoch_lin) * 0.1)
start_index_log = int(len(num_epoch_log) * 0.1)

# Adjusted data to exclude the initial phase
num_epoch_lin_adj = num_epoch_lin[start_index_lin:]
train_loss_lin_adj = train_loss_lin[start_index_lin:]
val_loss_lin_adj = val_loss_lin[start_index_lin:]

num_epoch_log_adj = num_epoch_log[start_index_log:]
train_loss_log_adj = train_loss_log[start_index_log:]
val_loss_log_adj = val_loss_log[start_index_log:]

# Find the minimum validation loss and corresponding epoch for each model
min_val_loss_lin = np.min(val_loss_lin_adj)
min_epoch_lin = num_epoch_lin_adj[np.argmin(val_loss_lin_adj)]
min_val_loss_log = np.min(val_loss_log_adj)
min_epoch_log = num_epoch_log_adj[np.argmin(val_loss_log_adj)]

# Create a figure with two subplots
fig, axs = plt.subplots(1, 2, figsize=(10.5 * 0.75, 5 * 0.75))  # 1 row, 2 columns

# Set common x-axis and y-axis labels for the subplots
for ax in axs:
    ax.set_xlabel(r'Epoch ($\times 10^3$)')
    ax.set_ylabel(r'$L(y, \hat{y})$')

# Plot data for the linear model
axs[0].plot(num_epoch_lin_adj / 1000, train_loss_lin_adj, color='blue', label='Training', alpha=0.7)
axs[0].plot(num_epoch_lin_adj / 1000, val_loss_lin_adj, color='orange', label='Validation', alpha=0.7)
axs[0].axvline(x=min_epoch_lin / 1000, color='black', linestyle='--', alpha=0.7)  # Mark epoch with min validation loss
axs[0].scatter(min_epoch_lin / 1000, min_val_loss_lin, color='red', s=25, zorder=5)  # Highlight min validation loss
axs[0].set_ylabel(r'$L_{\mathrm{MSE}}(y, \hat{y})$')  # Set specific y-label for linear model

# Plot data for the logarithmic model
axs[1].plot(num_epoch_log_adj / 1000, train_loss_log_adj, color='blue', label='Training', alpha=0.7)
axs[1].plot(num_epoch_log_adj / 1000, val_loss_log_adj, color='orange', label='Validation', alpha=0.7)
axs[1].axvline(x=min_epoch_log / 1000, color='black', linestyle='--', alpha=0.7)  # Mark epoch with min validation loss
axs[1].scatter(min_epoch_log / 1000, min_val_loss_log, color='red', s=25, zorder=5)  # Highlight min validation loss
axs[1].set_ylabel(r'$L_{\mathrm{MSLE}}(y, \hat{y})$')  # Set specific y-label for logarithmic model

# Create custom legend entries for minimum validation loss points
lin_1 = fr'min $L_{{\mathrm{{MSE}}}}(y, \hat{{y}}) = {sci_notation(min_val_loss_lin, sig_fig=2)}$'
log_1 = fr'min $L_{{\mathrm{{MSLE}}}}(y, \hat{{y}}) = {sci_notation(min_val_loss_log, sig_fig=2)}$'

# Custom legend markers for the minimum validation loss points
min_val_loss_marker_lin = Line2D([0], [0], marker='o', color='w', label=lin_1, markersize=5, markerfacecolor='red')
min_val_loss_marker_log = Line2D([0], [0], marker='o', color='w', label=log_1, markersize=5, markerfacecolor='red')

# Add individual legends to each subplot
axs[0].legend(handles=[min_val_loss_marker_lin], loc='upper center', fontsize=10)
axs[1].legend(handles=[min_val_loss_marker_log], loc='upper center', fontsize=10)

# Shared legend for Training and Validation
handles, labels = axs[0].get_legend_handles_labels()
unique_labels, unique_handles = [], []
for handle, label in zip(handles, labels):
    if label not in unique_labels:
        unique_labels.append(label)
        unique_handles.append(handle)

# Place the shared legend above the subplots
fig.legend(unique_handles, unique_labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.0))

# Adjust layout to fit everything neatly
fig.tight_layout(rect=[0, 0.03, 1, 0.95])

# Save the figure as a PDF file
output_dir = '../output/figures'
output_file_path = f'{output_dir}/iteration_process.pdf'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)  # Create output directory if it doesn't exist
plt.savefig(output_file_path, format='pdf')  # Save the plot as a PDF
plt.show()
