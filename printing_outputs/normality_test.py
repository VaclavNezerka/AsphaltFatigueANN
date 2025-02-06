import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import seaborn as sns
import os

# Enable LaTeX rendering for plots
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Load the transformed data from the specified Excel file
transformed_data_path = '../excel_spreadsheets/asphalt_20_deg_transformed.xlsx'
transformed_df = pd.read_excel(transformed_data_path)

# Extract data from specific columns and remove any missing values
data_cycles = transformed_df['Number of cycles (times)'].dropna().values  # Original data
data_boxcox = transformed_df['Number of cycles (Box-Cox transformed)'].dropna().values  # Box-Cox transformed data
data_ln = transformed_df['Number of cycles (Log transformed)'].dropna().values  # Logarithmic transformed data

# Initialize the plot with a larger figure size for better readability
plt.figure(figsize=(10, 8))

# Define custom x-labels in LaTeX format for each dataset
x_labels = {
    'Original data': r'$N_{\mathrm{f}}$',  # Original data (cycles)
    'Logarithmic transformation': r'$\ln{(N_{\mathrm{f}})}$',  # Log-transformed
    'Box-Cox transformation': r'$\frac{N_{\mathrm{f}}^{\lambda} - 1}{\lambda}$'  # Box-Cox-transformed
}

# Loop through datasets and plot Q-Q plots and histograms
for i, (data, label) in enumerate([
    (data_cycles, 'Original data'),
    (data_ln, 'Logarithmic transformation'),
    (data_boxcox, 'Box-Cox transformation')
]):
    # Q-Q Plot (left column)
    plt.subplot(3, 2, 2 * i + 1)
    stats.probplot(data, dist="norm", plot=plt)  # Generate Q-Q plot
    plt.title(rf'{label}')  # Set the title dynamically
    plt.xlabel(r'Theoretical quantiles')  # Standard x-label for Q-Q plots
    plt.ylabel(x_labels[label])  # Custom y-label based on the dataset

    # Histogram (right column)
    plt.subplot(3, 2, 2 * i + 2)
    sns.histplot(data, kde=False)  # Plot histogram without a KDE curve
    plt.title(rf'{label}')  # Set the title dynamically
    plt.xlabel(x_labels[label])  # Custom x-label based on the dataset
    plt.ylabel('Frequency')  # Standard y-label for histograms

    # Perform Shapiro-Wilk normality test
    stat, p_value = stats.shapiro(data)  # Returns statistic and p-value
    # Display the p-value in the plot (upper-right corner)
    plt.text(
        0.95, 0.95, rf'$p = {p_value:.6f}$',
        transform=plt.gca().transAxes,  # Use axis coordinates for placement
        verticalalignment='top', horizontalalignment='right'
    )

# Adjust layout and save
plt.tight_layout()
output_dir = '../output/figures'
output_file_path = f'{output_dir}/normality_test.pdf'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)  # Create output directory if it doesn't exist
plt.savefig(output_file_path, format='pdf')  # Save the plot as a PDF
plt.show()