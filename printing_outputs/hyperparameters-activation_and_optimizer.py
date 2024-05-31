import numpy as np
import matplotlib.pyplot as plt

# Enable LaTeX integration
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Define activations and optimizers
activations = ['sigmoid', 'sigmoid', 'sigmoid', 'ReLU', 'ReLU', 'ReLU', 'linear', 'linear', 'linear']
optimizers = ['Adam', 'Nadam', 'RMSprop', 'Adam', 'Nadam', 'RMSprop', 'Adam', 'Nadam', 'RMSprop']

# R^2 scores for linear and logarithmic models
log_R2 = [-0.046255235, -0.014857362, -0.132222508, 0.657752613, 0.652082208, 0.677268131, -0.065627827, -0.065595357, -0.065094595]
lin_R2 = [-0.083464374, -0.083541762, -0.083555622, 0.725979163, 0.668218538, 0.753222295, 0.282632966, 0.282628126, 0.282590981]

# Colors for optimizers
colors = ['r', 'g', 'b']

# Unique activations for grouping
unique_activations = sorted(set(activations), key=activations.index)

# Create a figure and a set of subplots
fig, axs = plt.subplots(1, 2, figsize=(10.5 * 0.75, 5 * 0.75))  # 1 row, 2 columns

bar_width = 0.25  # width of the bars
indices = np.arange(len(unique_activations))  # group positions

# Plotting bars for each optimizer
for i, optimizer in enumerate(sorted(set(optimizers), key=optimizers.index)):
    # Filter R2 scores by optimizer
    lin_r2_by_optimizer = [lin_R2[j] for j, opt in enumerate(optimizers) if opt == optimizer]
    log_r2_by_optimizer = [log_R2[j] for j, opt in enumerate(optimizers) if opt == optimizer]

    # Left subplot for linear RMSE criterion
    axs[0].bar(indices + i * bar_width - 0.5 * bar_width, lin_r2_by_optimizer, color=colors[i], width=bar_width, label=optimizer)

    # Right subplot for logarithmic RMSE criterion
    axs[1].bar(indices + i * bar_width - 0.5 * bar_width, log_r2_by_optimizer, color=colors[i], width=bar_width, label=optimizer)

# Adjusting the plots
for ax in axs:
    ax.set_ylabel(r'$\overline{R^2}$')
    ax.set_xticks(indices + bar_width / 2)  # Center the tick under the group of bars
    ax.set_xticklabels(unique_activations, ha="center")  # Set x-tick labels
    ax.axhline(0, color='black', linewidth=0.8)  # Add a line at y=0 for clarity

# Titles for subplots
axs[0].set_title(r'$L_{\mathrm{MSE}}(y, \hat{y})$')
axs[1].set_title(r'$L_{\mathrm{MSLE}}(y, \hat{y})$')

# Common legend
plt.subplots_adjust(top=0.95)
handles, labels = axs[0].get_legend_handles_labels()  # Assuming handles and labels are the same for both subplots
fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1))

# Adjust layout to fit the legend
fig.tight_layout(rect=[0, 0, 1, 0.90])

# Save the figure
output_file_path = '../output/figures/r2_score_different_activations_and_optimizers.pdf'
plt.savefig(output_file_path, format='pdf')
plt.show()