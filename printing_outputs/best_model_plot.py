import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Enable LaTeX integration for better text rendering in plots
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Define options for datasets based on temperatures
twenty_deg = '20_deg'

# Define options for loss optimization (either logarithmic or linear)
lin = 'lin'
log = 'log'

# Set the current configuration (temperature and optimization type)
actual_temp = twenty_deg
actual_opt = log

# Define the path to the Excel file containing hyperparameter tuning results
file_path = f'../excel_spreadsheets/{actual_opt}{actual_temp}_hyperparameter_tuning_results.xlsx'

# Load specific columns from the Excel file
# Columns include number of layers ('num_layers'), dense neurons ('dense'), and R^2 test scores ('r2_test')
df = pd.read_excel(file_path, usecols=['num_layers', 'dense', 'r2_test'])

# Extract unique values of 'dense' and 'num_layers' to assign marker types
dense_values = df['dense'].unique()
num_layer_values = df['num_layers'].unique()

# Define marker types for different combinations of dense and num_layers
marker_types = ['o', 's', '^', 'D', '*']  # Circle, Square, Triangle, Diamond, Star
dense_to_marker = {dense: marker for dense, marker in zip(dense_values, marker_types)}
num_layer_to_marker = {num_layer: marker for num_layer, marker in zip(num_layer_values, marker_types)}

# Define colors to differentiate between 'num_layers'
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

# Group the dataset by 'num_layers' and 'dense' and calculate mean and standard deviation of 'r2_test'
agg_df = df.groupby(['num_layers', 'dense'])['r2_test'].agg(['mean', 'std']).reset_index()

# Create a figure with 1 row and 2 subplots
fig, axs = plt.subplots(1, 2, figsize=(10.5 * 0.75, 6 * 0.75))  # Adjusting figure size for better visuals

# First subplot: Plot mean r2_test against num_layers
ax = axs[0]
for _, row in agg_df.iterrows():
    num_layers = row['num_layers']
    dense = row['dense']
    mean = row['mean']
    ax.scatter(
        num_layers, mean,
        label=r'$n_{\mathrm{neur,h}}=%d$, $n_{\mathrm{h}}=%d$' % (dense, num_layers),
        color=colors[int(num_layers) % len(colors) - 1],  # Assign color based on num_layers
        marker=dense_to_marker[dense],  # Assign marker based on dense value
        facecolors='none'  # Use unfilled markers for better visualization
    )

# Set labels and limits for the first subplot
ax.set_xlabel(r'$n_{\mathrm{h}}$')  # Label for num_layers
ax.set_ylabel(r'$\overline{R^2}$')  # Mean R^2 score
ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # Ensure x-axis has integer ticks
ax.set_ylim(0, 1)  # Set y-axis range from 0 to 1

# Second subplot: Plot mean r2_test against dense
ax = axs[1]
for _, row in agg_df.iterrows():
    num_layers = row['num_layers']
    dense = row['dense']
    mean = row['mean']
    ax.scatter(
        dense, mean,
        label=r'$n_{\mathrm{neur,h}}=%d$, $n_{\mathrm{h}}=%d$' % (dense, num_layers),
        color=colors[int(num_layers) % len(colors) - 1],  # Assign color based on num_layers
        marker=dense_to_marker[dense],  # Assign marker based on dense value
        facecolors='none'  # Use unfilled markers for better visualization
    )

# Set labels and limits for the second subplot
ax.set_xlabel(r'$n_{\mathrm{neur,h}}$')  # Label for dense
ax.set_ylabel(r'$\overline{R^2}$')  # Mean R^2 score
ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # Ensure x-axis has integer ticks
ax.set_ylim(0, 1)  # Set y-axis range from 0 to 1

# Add a common legend for the entire figure
plt.subplots_adjust(top=0.85)  # Adjust space at the top for the legend
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.0))  # Place legend at the top

# Adjust layout to ensure the figure fits well with the legend
fig.tight_layout(rect=[0, 0, 1, 0.75])

# Save the figure to a PDF file
output_dir = '../output/figures'
output_file_path = f'{output_dir}/{actual_opt}{actual_temp}_avg_r2_score.pdf'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)  # Create output directory if it doesn't exist
plt.savefig(output_file_path, format='pdf')  # Save the plot as a PDF
plt.show()  # Display the plot
