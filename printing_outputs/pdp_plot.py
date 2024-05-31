import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import joblib
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LogNorm
import pandas as pd


# Load the pretrained model
model = keras.models.load_model(
    '../output/lin/20_deg/2layers_123dense_3175epochs_adam_optimizer_relu_activation/final_model.h5')
output_file_path = (f'../output/figures/pdp_plot_')

# Load the data
file_path = '../output/lin/20_deg/2layers_123dense_3175epochs_adam_optimizer_relu_activation/asphalt_20_deg_predicted.xlsx'
data = pd.read_excel(file_path)

# Enable LaTeX integration
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Extract 'Air Voids' and 'Binder Content' columns
av = data['Air Voids (%)'].values
bc = data['Binder content (%)'].values

# Define actual optimization
lin = 'lin'
log = 'log'
actual_opt = lin

# Load the MinMaxScaler that was fit during training
scaler = joblib.load('../scaler_20_deg_with_project_n.save')

# Define the ranges for 'Binder content' and 'Air voids'
binder_content_range = np.linspace(4, 7, 400)
air_voids_range = np.linspace(1, 13, 400)

# Preset fixed strain values for each subplot
fixed_strains = [200, 400]

# Initialize global min and max values
global_min = np.inf
global_max = -np.inf

# Create a list to store reshaped predictions
temp_reshaped_predictions = []

# Create a 1x2 subplot structure
fig, axs = plt.subplots(1, 2, figsize=(10.5 * 0.65, 7 * 0.65))

# Flatten axes array for easy iteration
axs = axs.flatten()

# Define levels for contour plots
levels = np.logspace(np.log10(10000), np.log10(10000000), num=10)

# Loop through each fixed strain value and corresponding axis
for i, strain_fixed in enumerate(fixed_strains):
    # Generate a meshgrid for 'Binder content' and 'Air voids'
    binder_content, air_voids = np.meshgrid(binder_content_range, air_voids_range)

    # Flatten the meshgrid matrices
    binder_content_flat = binder_content.flatten()
    air_voids_flat = air_voids.flatten()

    # Create a repeated array for the current 'Strain'
    strain_fixed_array = np.full(binder_content_flat.shape, strain_fixed)

    # Stack the input parameters together for scaling
    model_inputs = np.stack([binder_content_flat, strain_fixed_array, air_voids_flat], axis=1)

    # Scale the inputs
    model_inputs_scaled = scaler.transform(model_inputs)

    # Predict the 'Number of cycles' using the model
    predicted_cycles_scaled = model.predict(model_inputs_scaled).flatten()

    # Apply a lower limit to the predictions
    predicted_cycles_scaled = np.clip(predicted_cycles_scaled, 10000, 10000000)

    # Reshape the predictions to match the meshgrid shape
    predicted_cycles_reshaped = predicted_cycles_scaled.reshape(binder_content.shape)

    # Apply logarithmic transformation for better visualization
    predicted_cycles_reshaped_log = np.log(predicted_cycles_reshaped + 1)

    # Back-transform the predictions to original scale
    predicted_cycles_back_transformed = np.exp(predicted_cycles_reshaped_log) - 1

    # Store the reshaped predictions
    temp_reshaped_predictions.append(predicted_cycles_reshaped_log)

    # Update global min and max values
    global_min = min(global_min, predicted_cycles_back_transformed.min())
    global_max = max(global_max, predicted_cycles_back_transformed.max())

    # Plotting
    ax = axs[i]
    contourf = ax.contourf(air_voids, binder_content, predicted_cycles_back_transformed,
                           levels=levels, cmap='jet', norm=LogNorm(vmin=levels.min(), vmax=levels.max()))

    contour = ax.contour(air_voids, binder_content, predicted_cycles_back_transformed,
                         levels=levels, colors='k')
    ax.scatter(av, bc, color='white', marker='v', s=35, edgecolors='black', label='Data Points')

    # Titles and labels
    times_10 = r'\times 10^{-6}$'
    nf = r'$N_{\mathrm{f}}$'
    ax.set_title(rf'$\varepsilon = {strain_fixed} {times_10}', fontsize=10)
    ax.set_xlabel(r'Air voids (\%)')
    ax.set_ylabel(r'Binder content (\%)')

    # Set integer ticks for x and y axes
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

# Add a common colorbar for all subplots
plt.subplots_adjust(top=0.8, right=0.8)
cbar_ax = fig.add_axes([0.15, 0.95, 0.7, 0.04])
fig.colorbar(contourf, cax=cbar_ax, orientation='horizontal', fraction=0.02, pad=0.05, label=r'$N_{\mathrm{f}}$')

# Adjust layout and save the figure
plt.tight_layout(rect=[0, 0, 1, 0.9])
plt.savefig(output_file_path + actual_opt + '.pdf', format='pdf')
plt.show()