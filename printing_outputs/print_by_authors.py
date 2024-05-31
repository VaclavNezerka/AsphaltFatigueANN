import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from matplotlib.lines import Line2D

# File paths
lin_predicted_table_path = '../output/lin/20_deg/2layers_123dense_3175epochs_adam_optimizer_relu_activation/asphalt_20_deg_predicted.xlsx'
log_predicted_table_path = '../output/log/20_deg/1layers_10dense_2001epochs_adam_optimizer_relu_activation/asphalt_20_deg_predicted.xlsx'
output_file_path = (f'../output/figures/print_by_authors.pdf')

# Configure matplotlib for LaTeX-style text rendering
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Load data for both linear and logarithmic models
df_train_lin = pd.read_excel(lin_predicted_table_path, sheet_name='fatigue data - train')
df_test_lin = pd.read_excel(lin_predicted_table_path, sheet_name='fatigue data - validate')
df_train_log = pd.read_excel(log_predicted_table_path, sheet_name='fatigue data - train')
df_test_log = pd.read_excel(log_predicted_table_path, sheet_name='fatigue data - validate')

# Extract 'predictions' and 'outputs' columns as arrays for each DataFrame
train_predictions_lin = df_train_lin['Predicted'].values
train_outputs_lin = df_train_lin['Number of cycles (times)'].values
test_predictions_lin = df_test_lin['Predicted'].values
test_outputs_lin = df_test_lin['Number of cycles (times)'].values

train_predictions_log = df_train_log['Predicted'].values
train_outputs_log = df_train_log['Number of cycles (times)'].values
test_predictions_log = df_test_log['Predicted'].values
test_outputs_log = df_test_log['Number of cycles (times)'].values

# Load all sheets to extract author information
all_sheets_lin = pd.read_excel(lin_predicted_table_path, sheet_name=None)
all_sheets_log = pd.read_excel(log_predicted_table_path, sheet_name=None)

# Combine train and test data for comprehensive analysis
all_outputs_lin = np.concatenate([train_outputs_lin, test_outputs_lin])
all_predictions_lin = np.concatenate([train_predictions_lin, test_predictions_lin])
all_outputs_log = np.concatenate([train_outputs_log, test_outputs_log])
all_predictions_log = np.concatenate([train_predictions_log, test_predictions_log])

# Ensure outputs and predictions are squeezed to 1D arrays
all_outputs_lin = np.squeeze(np.array(all_outputs_lin))
all_predictions_lin = np.squeeze(np.array(all_predictions_lin))
all_outputs_log = np.squeeze(np.array(all_outputs_log))
all_predictions_log = np.squeeze(np.array(all_predictions_log))

# Determine min and max values for plotting
min_value_lin = np.min([all_outputs_lin, all_predictions_lin])
max_value_lin = np.max([all_outputs_lin, all_predictions_lin])
min_value_log = np.min([all_outputs_log, all_predictions_log])
max_value_log = np.max([all_outputs_log, all_predictions_log])

# Calculate R^2 scores
r2_lin = r2_score(all_outputs_lin, all_predictions_lin)
r2_log = r2_score(all_outputs_log, all_predictions_log)

# Extract unique authors from all sheets
authors_lin = pd.concat([df['Author'] for df in all_sheets_lin.values()]).unique()
authors_log = pd.concat([df['Author'] for df in all_sheets_log.values()]).unique()
all_authors = np.unique(np.concatenate((authors_lin, authors_log)))

# Encode authors to numeric values
le = LabelEncoder()
le.fit(all_authors)

# Define specific colors for authors
specific_colors = ['blue', 'green', 'orange', 'red']
color_map = {author: specific_colors[i % len(specific_colors)] for i, author in enumerate(all_authors)}

# Create a figure and a set of subplots
fig, axs = plt.subplots(1, 2, figsize=(10.5 * 0.75, 6.8 * 0.75))  # 1 row, 2 columns

# Configure the first subplot (Linear model)
axs[0].grid(True, linestyle='--', alpha=0.7)
axs[0].set_xscale('log')
axs[0].set_yscale('log')
axs[0].set_xlabel(r'$N_{\mathrm{f}}^{\mathrm{true}}$')
axs[0].set_ylabel(r'$N_{\mathrm{f}}^{\mathrm{predicted}}$', labelpad=-5)
axs[0].plot([min_value_lin, max_value_lin], [min_value_lin, max_value_lin], color='black')
axs[0].set_aspect('equal', adjustable='box')
axs[0].axis([min_value_lin, max_value_lin, min_value_lin, max_value_lin])
axs[0].text(0.05, 0.95, f'$R^2 = {r2_lin:.3f}$', transform=axs[0].transAxes,
            verticalalignment='top', horizontalalignment='left', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

# Configure the second subplot (Logarithmic model)
axs[1].grid(True, linestyle='--', alpha=0.7)
axs[1].set_xscale('log')
axs[1].set_yscale('log')
axs[1].set_xlabel(r'$N_{\mathrm{f}}^{\mathrm{true}}$')
axs[1].set_ylabel(r'$N_{\mathrm{f}}^{\mathrm{predicted}}$', labelpad=-5)
axs[1].plot([min_value_log, max_value_log], [min_value_log, max_value_log], color='black')
axs[1].set_aspect('equal', adjustable='box')
axs[1].axis([min_value_log, max_value_log, min_value_log, max_value_log])
axs[1].text(0.05, 0.95, f'$R^2 = {r2_log:.3f}$', transform=axs[1].transAxes,
            verticalalignment='top', horizontalalignment='left', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

# Set titles for the subplots
axs[0].set_title(r'$L_{\mathrm{MSE}}(y, \hat{y})$')
axs[1].set_title(r'$L_{\mathrm{MSLE}}(y, \hat{y})$')

# Plot data for the linear model
for sheet_name, df in all_sheets_lin.items():
    author_colors = df['Author'].map(color_map)
    axs[0].scatter(df['Number of cycles (times)'], df['Predicted'], color=author_colors, label=sheet_name)

# Plot data for the logarithmic model
for sheet_name, df in all_sheets_log.items():
    author_colors = df['Author'].map(color_map)
    axs[1].scatter(df['Number of cycles (times)'], df['Predicted'], color=author_colors, label=sheet_name)

# Adjust layout and add legend
plt.subplots_adjust(wspace=0.01)
legend_handles = [Line2D([0], [0], marker='o', color='none', label=author,
                         markerfacecolor=color, markeredgecolor=color, markersize=5) for author, color in color_map.items()]

fig.tight_layout(rect=[0, 0.00, 1, 1])
fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 1), ncol=len(legend_handles)//2)

# Save the figure
plt.savefig(output_file_path, format='pdf')
plt.show()