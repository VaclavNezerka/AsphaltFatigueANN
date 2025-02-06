import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import joblib
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LogNorm
import pandas as pd
from scipy.spatial import ConvexHull
from matplotlib.path import Path
import os

# Enable LaTeX rendering for better plot text formatting
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Load the pre-trained model
model = keras.models.load_model('../output/lin/20_deg/1kf_2layers_200dense_28epochs_adam_optimizer_relu_activation/best_model.h5')

# Load the dataset
file_path = '../output/lin/20_deg/1kf_2layers_200dense_28epochs_adam_optimizer_relu_activation/asphalt_20_deg_filtered_bc.xlsx'
data = pd.read_excel(file_path)

# Extract columns for Air Voids and Binder Content
av = data['Air Voids (%)'].values  # Air voids values
bc = data['Binder content (%)'].values  # Binder content values

# Load the scaler used during model training for data normalization
scaler = joblib.load('../scaler_20_deg_with_project_n.save')

# Define ranges for Binder Content and Air Voids for the meshgrid
binder_content_range = np.linspace(4, 6.7, 40)
air_voids_range = np.linspace(1, 13, 40)

# Define fixed strain values for each subplot
fixed_strains = [200, 400]

# Initialize global min and max values for color scaling
global_min = np.inf
global_max = -np.inf

# Create a 1x2 subplot layout
fig, axs = plt.subplots(1, 2, figsize=(10.5 * 0.65, 7 * 0.65))
axs = axs.flatten()

# Compute the convex hull of the measured data points
points = np.column_stack((av, bc))
hull = ConvexHull(points)
hull_path = Path(points[hull.vertices])  # Define the convex hull as a path

# Loop through each strain value and corresponding axis
for i, strain_fixed in enumerate(fixed_strains):
    # Generate a meshgrid for Binder Content and Air Voids
    binder_content, air_voids = np.meshgrid(binder_content_range, air_voids_range)

    # Flatten the meshgrid to create a list of inputs
    binder_content_flat = binder_content.flatten()
    air_voids_flat = air_voids.flatten()

    # Create an array with the fixed strain value
    strain_fixed_array = np.full(binder_content_flat.shape, strain_fixed)

    # Combine inputs into a single array and scale them
    model_inputs = np.stack([binder_content_flat, strain_fixed_array, air_voids_flat], axis=1)
    model_inputs_scaled = scaler.transform(model_inputs)

    # Predict the Number of Cycles using the model
    predicted_cycles_scaled = model.predict(model_inputs_scaled).flatten()
    predicted_cycles_scaled = np.clip(predicted_cycles_scaled, 10000, 10000000)  # Clip predictions to valid range
    predicted_cycles_reshaped = predicted_cycles_scaled.reshape(binder_content.shape)

    # Mask out predictions outside the convex hull
    mesh_points = np.column_stack((air_voids_flat, binder_content_flat))
    mask = ~hull_path.contains_points(mesh_points)  # Identify points outside the hull
    mask = mask.reshape(binder_content.shape)  # Reshape to match the meshgrid
    predicted_cycles_reshaped[mask] = np.nan  # Assign NaN to invalid regions

    # Plot the predictions using pcolormesh
    ax = axs[i]
    pcm = ax.pcolormesh(air_voids, binder_content, predicted_cycles_reshaped,
                        shading='auto', cmap='jet', norm=LogNorm(vmin=10000, vmax=10000000))

    # Overlay the convex hull as a green line
    for simplex in hull.simplices:
        ax.plot(points[simplex, 0], points[simplex, 1], 'g-', linewidth=0.5)

    # Scatter plot of measured data points
    ax.scatter(av, bc, color='white', marker='v', s=20, edgecolors='black', label='Data Points')

    # Add titles and labels
    ax.set_title(rf'$\varepsilon = {strain_fixed} \times 10^{{-6}}$', fontsize=10)
    ax.set_xlabel(r'Air voids (\%)')
    ax.set_ylabel(r'Binder content (\%)')

    # Ensure integer ticks for axes
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

# Add a common colorbar above the plots
plt.subplots_adjust(top=0.8, right=0.8)  # Adjust the layout for the colorbar
cbar_ax = fig.add_axes([0.15, 0.95, 0.7, 0.04])  # Define colorbar position
fig.colorbar(pcm, cax=cbar_ax, orientation='horizontal', label=r'$N_{\mathrm{f}}$')

# Enable grid for better readability
plt.grid(True, linestyle='--', alpha=0.7, linewidth=0.8)

# Final adjustments and save the plot
plt.tight_layout(rect=[0, 0, 1, 0.9])  # Adjust the layout for spacing

# Save the plot as a PDF file
output_dir = '../output/figures'
output_file_path = f'{output_dir}/pdp.pdf'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)  # Create output directory if it doesn't exist
plt.savefig(output_file_path, format='pdf')  # Save the plot as a PDF
plt.show()