import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr
import seaborn as sns
import os


# Function to capitalize material names conditionally
def conditional_capitalize(name):
    """
    Capitalize material names unless they are already uppercase.
    """
    if name.isupper():
        return name
    else:
        return name.capitalize()

# Function to calculate and display Spearman correlation on scatter plots
def spearman_annotate_box(x, y, ax):
    """
    Adds a Spearman correlation coefficient annotation to a scatter plot.
    """
    r, _ = spearmanr(x, y)  # Calculate Spearman correlation
    ax.text(0.95, 0.95, f'$\\rho$ = {r:.2f}',  # Add annotation box
            transform=ax.transAxes, fontsize=10,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle="round,pad=0.2"),
            verticalalignment='top', horizontalalignment='right')

# Enable LaTeX rendering for plots
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Load the dataset
file_path = "../excel_spreadsheets/asphalt_20_deg_filtered_bc.xlsx"
df = pd.read_excel(file_path, sheet_name='Sheet1')

# Extract material names from the 'Author' column and clean them
df['Material'] = df['Author'].apply(lambda x: str(x)[:-6] if isinstance(x, str) else 'Unknown')
df['Material'] = df['Material'].apply(conditional_capitalize)

# Recalculate the sample count for each material
material_counts = df['Material'].value_counts().to_dict()

# Append sample counts to material names
df['Material_with_counts'] = df['Material'].apply(lambda x: f"{x} ({material_counts[x]})")

# Define the features to include in the pairplot
selected_features = [
    'Binder content (%)',
    'Air Voids (%)',
    'Initial strain (µɛ)',
    'Number of cycles (times)'
]

renamed_features = {
    'Binder content (%)': r'Binder content (\%)',
    'Air Voids (%)': r'Air voids (\%)',
    'Initial strain (µɛ)': r"$\varepsilon \times 10^{-6}$",
    'Number of cycles (times)': r'$N_{\mathrm{f}}\times 10^{6}$'
}

# Check if all selected features exist in the dataset
missing_features = [feat for feat in selected_features if feat not in df.columns]
if missing_features:
    print(f"The following selected features are missing in the data: {missing_features}")

# Create a new DataFrame with the selected features and the annotated material names
data = df[selected_features].copy()
data['Material_with_counts'] = df['Material_with_counts']

# Normalize the 'Number of cycles (times)' column by dividing it by 1 million
data['Number of cycles (times)'] = data['Number of cycles (times)'] / 1_000_000

# Set the size of the plot to 0.9 times the dimensions of an A4 paper
plt.figure(figsize=(8.27 * 0.9, 11.69 * 0.9))

# Define new axis labels for better readability using LaTeX formatting
new_labels = [
    r'Binder content (\%)',
    r'Air voids (\%)',
    r"$\varepsilon \times 10^{-6}$",
    r'$N_{\mathrm{f}}\times 10^{6}$'
]

# Rename the columns in the DataFrame for plotting
data_renamed = data.rename(columns=renamed_features)

# Create the pairplot with scatter plots and histograms
pair_plot = sns.pairplot(
    data_renamed,
    hue='Material_with_counts',  # Differentiate by material with counts
    palette='muted',  # Use muted color palette
    plot_kws={'s': 30, 'alpha': 0.7},  # Scatter plot settings
    diag_kind='hist'  # Use histograms on the diagonal
)

# Annotate scatter plots with Spearman correlation coefficients
for i in range(len(selected_features)):
    for j in range(len(selected_features)):
        # Skip diagonal plots (histograms)
        if i != j:
            spearman_annotate_box(data[selected_features[j]], data[selected_features[i]], pair_plot.axes[i, j])

# Update axis labels with LaTeX formatting
for i, ax in enumerate(pair_plot.axes[-1, :]):
    ax.set_xlabel(new_labels[i])  # Set x-labels for the last row

for i, ax in enumerate(pair_plot.axes[:, 0]):
    ax.set_ylabel(new_labels[i])  # Set y-labels for the first column

# Adjust the legend to fit the plot
pair_plot._legend.set_title('Authors (sample count)')
pair_plot._legend.set_bbox_to_anchor((0.90, 0.5))  # Place the legend on the right
pair_plot._legend.borderaxespad = 0.
pair_plot.fig.subplots_adjust(right=0.75)  # Adjust plot spacing for the legend

# Save the plot as a PDF file
output_dir = '../output/figures'
output_file_path = f'{output_dir}/pairPlot.pdf'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)  # Create output directory if it doesn't exist
plt.savefig(output_file_path, format='pdf')  # Save the plot as a PDF
plt.show()