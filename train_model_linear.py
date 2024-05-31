import pandas as pd
import keras
import time
import shutil
from aux_functions.excel_modification import *
from aux_functions.train_ml_model import *
from aux_functions.print_automatic import *

# Defining options for datasets based on temperatures
twenty_deg = '20_deg'

# Defining options for loss optimization (either logarithmic or linear)
lin = 'lin'
log = 'log'

# Setting the currently used temperature and optimization type
actual_temp = twenty_deg
actual_opt = lin

# Establishing base paths for files and naming conventions
base_path = 'excel_spreadsheets/'
sample_name = 'asphalt'
file_name = base_path + sample_name + '_' + actual_temp

# Defining filenames for each processing step of the dataset
xls_separated = file_name + '_separated.xlsx'
xls_imputed = file_name + '_imputed.xlsx'
xls_reduced = file_name + '_reduced.xlsx'
xls_rounded = file_name + '_rounded.xlsx'
xls_filtered = file_name + '_filtered.xlsx'
xls_divided = file_name + '_divided.xlsx'
xls_predicted = file_name + '_predicted.xlsx'

# Processing steps for data preparation
separate_data(file_name + '.xlsx', xls_separated)  # Separates data into individual Excel files
drop_rows_with_missing_values(xls_separated, xls_reduced)  # Removes rows with any missing values
round_data_values(xls_reduced, xls_rounded)  # Rounds data values in Excel file
filter_outliers_by_quantile(xls_rounded, xls_filtered, 0.03, 0.90)  # Filters outliers by quantile
filter_outliers_by_zscore(xls_filtered, xls_filtered, threshold=3)  # Filters outliers using Z-score method
split_data_train_validate(xls_filtered, xls_divided)  # Splits data into training and validation sets

# Load the divided data
df_train_imputed = pd.read_excel(xls_divided, sheet_name='fatigue data - train')
df_validate_imputed = pd.read_excel(xls_divided, sheet_name='fatigue data - validate')

# Read Excel file with predictions
df = pd.read_excel(xls_filtered)

# Define input and output columns
input_columns = ['Binder content (%)', 'Initial strain (µɛ)', 'Air Voids (%)']
output_column = 'Number of cycles (times)'

# Feature scaling for input features
scaler = MinMaxScaler()
train_inputs = scaler.fit_transform(df_train_imputed[input_columns])
validate_inputs = scaler.transform(df_validate_imputed[input_columns])

# Extracting output values
train_outputs = df_train_imputed[output_column].values
validate_outputs = df_validate_imputed[output_column].values

# Start timer for training duration
start_time = time.time()
print(time.strftime("%H:%M:%S"))

# Training the model
dir_name, model, train_predictions, train_outputs, validate_predictions, validate_outputs = \
    train_ann_lin_val_train(
        actual_opt=actual_opt,
        actual_temp=actual_temp,
        train_inputs=train_inputs,
        train_outputs=train_outputs,
        validate_inputs=validate_inputs,
        validate_outputs=validate_outputs,
        num_layers=2,
        n_epochs=3175,
        dense=123,
        optimizer='adam',
        activation_function='relu'
    )

# Adding predictions to the table
add_predictions_to_table(xls_divided, xls_predicted, train_predictions, validate_predictions)

# Copying relevant files to the model directory
for xls_file in [xls_separated, xls_rounded, xls_filtered, xls_divided, xls_predicted]:
    new_file_path = os.path.join(dir_name, os.path.basename(xls_file))
    shutil.copy(xls_file, new_file_path)

# Calculate and log the training duration
end_time = time.time()
training_duration = end_time - start_time
print(f"Training duration: {training_duration:.0f} seconds")
