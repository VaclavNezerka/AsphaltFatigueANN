import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from matplotlib.lines import Line2D
from scipy.stats import spearmanr, pearsonr
import os

# Function to calculate adjusted R^2
def adjusted_r2(r2, n, p):
    """
    Compute the adjusted R^2 given R^2, number of samples (n), and predictors (p).
    """
    return 1 - (1 - r2) * ((n - 1) / (n - p - 1))

# Function to calculate Scatter Index (SI)
def scatter_index(true, predicted):
    """
    Compute the scatter index (SI) which is normalized RMSE.
    """
    return np.sqrt(np.mean((predicted - true) ** 2)) / np.mean(true)

# Configure matplotlib for LaTeX-style rendering
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# File path for predicted data
lin_predicted_table_path = '../output/lin/20_deg/1kf_2layers_200dense_28epochs_adam_optimizer_relu_activation/asphalt_20_deg_predicted.xlsx'

# Load training and testing data for the linear model
df_train_lin = pd.read_excel(lin_predicted_table_path, sheet_name='Train')
df_test_lin = pd.read_excel(lin_predicted_table_path, sheet_name='Test')

# Extract predictions and actual values as arrays
train_predictions_lin = df_train_lin['Predicted'].values
train_outputs_lin = df_train_lin['Actual'].values
test_predictions_lin = df_test_lin['Predicted'].values
test_outputs_lin = df_test_lin['Actual'].values

# Combine all sheets to extract author information
all_sheets_lin = pd.read_excel(lin_predicted_table_path, sheet_name=None)

# Concatenate training and testing data for overall analysis
all_outputs_lin = np.concatenate([train_outputs_lin, test_outputs_lin])
all_predictions_lin = np.concatenate([train_predictions_lin, test_predictions_lin])

# Ensure data arrays are 1D
all_outputs_lin = np.squeeze(np.array(all_outputs_lin))
all_predictions_lin = np.squeeze(np.array(all_predictions_lin))

# Determine min and max values for plotting
min_value_lin = np.min([all_outputs_lin, all_predictions_lin])
max_value_lin = np.max([all_outputs_lin, all_predictions_lin])

# Calculate performance metrics
r2_lin = r2_score(all_outputs_lin, all_predictions_lin)
adj_r2_lin = adjusted_r2(r2_lin, len(all_outputs_lin), 3)  # Adjusted R^2 (assuming 3 predictors)
si_lin = scatter_index(all_outputs_lin, all_predictions_lin)
pearson_lin, _ = pearsonr(all_outputs_lin, all_predictions_lin)
spearman_lin, _ = spearmanr(all_outputs_lin, all_predictions_lin)

# Extract and encode unique author names
authors_lin = pd.concat([df['Author'] for df in all_sheets_lin.values()]).unique()
all_authors = np.unique(authors_lin)

# Encode authors into numeric values for mapping
le = LabelEncoder()
le.fit(all_authors)

# Assign specific colors for each author
specific_colors = ['blue', 'green', 'orange', 'red', 'purple', 'yellow', 'cyan', 'magenta']
color_map = {author: specific_colors[i % len(specific_colors)] for i, author in enumerate(all_authors)}

# Create a single plot for true vs. predicted values
fig, ax = plt.subplots(1, 1, figsize=(10.5 * 0.75, 6.8 * 0.75))

# Configure the plot (linear model)
ax.grid(True, linestyle='--', alpha=0.7)
ax.set_xscale('log')  # Logarithmic scale for x-axis
ax.set_yscale('log')  # Logarithmic scale for y-axis
ax.set_xlabel(r'$N_{\mathrm{f}}^{\mathrm{true}}$')  # X-axis label
ax.set_ylabel(r'$N_{\mathrm{f}}^{\mathrm{predicted}}$', labelpad=-5)  # Y-axis label
ax.plot([min_value_lin, max_value_lin], [min_value_lin, max_value_lin], color='black')  # Diagonal line (y=x)
ax.set_aspect('equal', adjustable='box')
ax.axis([min_value_lin, max_value_lin, min_value_lin, max_value_lin])  # Axis limits

# Display metrics on the plot
ax.text(0.05, 0.95, f'$R^2 = {r2_lin:.3f}$\n'
                    f'$R^2_{{\\mathrm{{adj}}}} = {adj_r2_lin:.3f}$\n'
                    f'SI = {si_lin:.3f}',
        transform=ax.transAxes, verticalalignment='top',
        horizontalalignment='left', fontsize=10,
        bbox=dict(facecolor='white', alpha=0.5))

# Title for the plot
ax.set_title(r'$L_{\mathrm{MSE}}(y, \hat{y})$')

# Scatter plot of true vs. predicted values, color-coded by author
for sheet_name, df in all_sheets_lin.items():
    author_colors = df['Author'].map(color_map)
    ax.scatter(df['Actual'], df['Predicted'], color=author_colors, label=sheet_name)

# Add legend with custom author colors
legend_handles = [Line2D([0], [0], marker='o', color='none', label=author,
                         markerfacecolor=color, markeredgecolor=color, markersize=5) for author, color in color_map.items()]

# Adjust layout and add legend
fig.tight_layout(rect=[0, 0, 1, 0.9])
fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=len(legend_handles) // 2,
           fontsize=9, markerscale=1.00)

# Save the plot as a PDF file
output_dir = '../output/figures'
output_file_path = f'{output_dir}/true_vs_prediction.pdf'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)  # Create output directory if it doesn't exist
plt.savefig(output_file_path, format='pdf')  # Save the plot as a PDF
plt.show()
