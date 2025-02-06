import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import joblib
import pandas as pd
import os

# Enable LaTeX rendering for better plot text formatting
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Load the pre-trained model
model = keras.models.load_model('../output/lin/20_deg/1kf_2layers_200dense_28epochs_adam_optimizer_relu_activation/best_model.h5')

# Load the dataset
file_path = '../output/lin/20_deg/1kf_2layers_200dense_28epochs_adam_optimizer_relu_activation/asphalt_20_deg_filtered_bc.xlsx'
data = pd.read_excel(file_path)

# Load the MinMaxScaler that was used during model training for scaling the input data
scaler = joblib.load('../scaler_20_deg_with_project_n.save')

# Define air void levels and corresponding binder content ranges for each level (manually)
air_voids_levels = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # Air void levels
bc_ranges = {
    2: np.linspace(4.8, 6.2, 40),  # Binder content range for AV=2%
    3: np.linspace(4.5, 6.5, 40),
    4: np.linspace(4.3, 6.8, 40),
    5: np.linspace(4.2, 6.7, 40),
    6: np.linspace(4.1, 6.6, 40),
    7: np.linspace(4.1, 6.3, 40),
    8: np.linspace(4.1, 6.1, 40),
    9: np.linspace(4.2, 5.8, 40),
    10: np.linspace(4.3, 5.6, 40),
    11: np.linspace(4.4, 5.4, 40),
    12: np.linspace(4.8, 5.2, 40)  # Binder content range for AV=12%
}

# Preset a fixed strain value for the analysis
strain_fixed = 200

# Initialize the plot with a suitable figure size
plt.figure(figsize=(10.5 * 0.75, 6.8 * 0.75))

# Define line styles for differentiating between air void levels
line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':']

# Loop through each air void level and its corresponding binder content range
for i, air_voids_value in enumerate(air_voids_levels):
    # Create input arrays for the current air void level
    binder_content_flat = bc_ranges[air_voids_value]  # Binder content range
    air_voids_flat = np.full(binder_content_flat.shape, air_voids_value)  # Air void level
    strain_fixed_array = np.full(binder_content_flat.shape, strain_fixed)  # Fixed strain value

    # Prepare inputs for the model by stacking them together
    model_inputs = np.stack([binder_content_flat, strain_fixed_array, air_voids_flat], axis=1)
    model_inputs_scaled = scaler.transform(model_inputs)  # Scale the inputs

    # Predict the 'Number of cycles' using the model
    predicted_cycles_scaled = model.predict(model_inputs_scaled).flatten()
    predicted_cycles_scaled = np.clip(predicted_cycles_scaled, 770, 20000000)  # Clip predictions to valid range

    # Plot the predictions with a unique line style for each air void level
    plt.plot(binder_content_flat, predicted_cycles_scaled, line_styles[i % len(line_styles)],
             label=f'AV = {air_voids_value}\%', linewidth=1.5)

# Set axis labels
plt.xlabel('Binder content (\%)')
plt.ylabel(r'$N_{\mathrm{f}}$')

# Use logarithmic scale for the y-axis
plt.yscale('log')

# Add a legend for air void levels
plt.legend(title='Air Voids (\%)', fontsize=10)

# Add a grid for better visualization
plt.grid(True)

# Adjust layout to avoid overlap
plt.tight_layout()

# Save the plot as a PDF file
output_dir = '../output/figures'
output_file_path = f'{output_dir}/pdp_2d.pdf'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)  # Create output directory if it doesn't exist
plt.savefig(output_file_path, format='pdf')  # Save the plot as a PDF
plt.show()
