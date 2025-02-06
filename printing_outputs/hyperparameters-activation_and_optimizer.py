import numpy as np
import matplotlib.pyplot as plt
import os

# Enable LaTeX integration for better text rendering in plots
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Define activation functions and optimizers for grouping
activations = ['sigmoid', 'sigmoid', 'sigmoid', 'ReLU', 'ReLU', 'ReLU', 'linear', 'linear', 'linear']
optimizers = ['Adam', 'Nadam', 'RMSprop', 'Adam', 'Nadam', 'RMSprop', 'Adam', 'Nadam', 'RMSprop']

# R^2 scores for linear and logarithmic criteria (copied from xlsx file manually)
log_R2 = [0.046255235, 0.014857362, 0.132222508, 0.722752613, 0.710082208, 0.730268131, -0.045627827, -0.045595357, -0.045094595]
lin_R2 = [0.083464374, 0.083541762, 0.083555622, 0.755979163, 0.728218538, 0.813222295, 0.342632966, 0.342628126, 0.342590981]

# Define colors for each optimizer
colors = ['r', 'g', 'b']  # Red, Green, Blue

# Extract unique activations while preserving their order
unique_activations = sorted(set(activations), key=activations.index)

# Create a figure with 1 row and 2 subplots
fig, axs = plt.subplots(1, 2, figsize=(10.5 * 0.75, 5 * 0.75))  # Adjust the figure size for better readability

# Define bar width and indices for grouped bars
bar_width = 0.25
indices = np.arange(len(unique_activations))  # Group positions for activations

# Iterate through optimizers and plot bars for linear and logarithmic R^2 scores
for i, optimizer in enumerate(sorted(set(optimizers), key=optimizers.index)):
    # Filter R^2 scores by optimizer
    lin_r2_by_optimizer = [lin_R2[j] for j, opt in enumerate(optimizers) if opt == optimizer]
    log_r2_by_optimizer = [log_R2[j] for j, opt in enumerate(optimizers) if opt == optimizer]

    # Left subplot: Linear RMSE criterion
    axs[0].bar(
        indices + i * bar_width - 0.5 * bar_width,  # Adjust bar positions
        lin_r2_by_optimizer,  # Linear R^2 scores for current optimizer
        color=colors[i], width=bar_width, label=optimizer  # Set bar color and label
    )

    # Right subplot: Logarithmic RMSE criterion
    axs[1].bar(
        indices + i * bar_width - 0.5 * bar_width,  # Adjust bar positions
        log_r2_by_optimizer,  # Logarithmic R^2 scores for current optimizer
        color=colors[i], width=bar_width, label=optimizer  # Set bar color and label
    )

# Configure each subplot with labels, titles, and formatting
for ax in axs:
    ax.set_ylabel(r'$\overline{R^2}$')  # Set y-axis label
    ax.set_xticks(indices + bar_width / 2)  # Center the tick under the group of bars
    ax.set_xticklabels(unique_activations, ha="center")  # Set x-tick labels to activation names
    ax.axhline(0, color='black', linewidth=0.8)  # Add a horizontal line at y=0 for clarity

# Add titles for subplots
axs[0].set_title(r'$L_{\mathrm{MSE}}(y, \hat{y})$')  # Linear RMSE criterion
axs[1].set_title(r'$L_{\mathrm{MSLE}}(y, \hat{y})$')  # Logarithmic RMSE criterion

# Add a common legend for the entire figure
plt.subplots_adjust(top=0.95)  # Adjust top margin for the legend
handles, labels = axs[0].get_legend_handles_labels()  # Retrieve legend handles and labels
fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1))  # Place the legend at the top

# Adjust layout to accommodate the legend
fig.tight_layout(rect=[0, 0, 1, 0.90])

# Save the figure to a PDF file
output_dir = '../output/figures'
output_file_path = f'{output_dir}/2_score_different_activations_and_optimizers.pdf'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)  # Create output directory if it doesn't exist
plt.savefig(output_file_path, format='pdf')  # Save the plot as a PDF
plt.show()  # Display the plot