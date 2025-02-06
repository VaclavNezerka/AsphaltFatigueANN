import time
import shutil
import pandas as pd
from aux_functions.excel_modification import *
from aux_functions.train_ml_model import *
from aux_functions.metrics_calculation import *
import keras

# --------------------------------------------
# CONFIGURATION AND PATH DEFINITIONS
# --------------------------------------------

# Dataset temperature options
twenty_deg = '20_deg'

# Loss optimization options (linear/logarithmic)
lin = 'lin'
log = 'log'

# Set the current temperature and optimization type
actual_temp = twenty_deg
actual_opt = log

# Base path for files and dataset naming
base_path = 'excel_spreadsheets/'
sample_name = 'asphalt'
file_name = base_path + sample_name + '_' + actual_temp

# Define filenames for different steps in the dataset preparation process
xls_separated = file_name + '_separated.xlsx'
xls_reduced = file_name + '_reduced.xlsx'
xls_rounded = file_name + '_rounded.xlsx'
xls_transformed = file_name + '_transformed.xlsx'
xls_filtered_bc = file_name + '_filtered_bc.xlsx'
xls_filtered_ln = file_name + '_filtered_ln.xlsx'
xls_divided = file_name + '_divided.xlsx'
xls_predicted = file_name + '_predicted.xlsx'

# --------------------------------------------
# DATA PREPARATION
# --------------------------------------------

# Step 1: Separate data into individual Excel files
separate_data(file_name + '.xlsx', xls_separated)

# Step 2: Drop rows with any missing values
drop_rows_with_missing_values(xls_separated, xls_reduced)

# Step 3: Round values in specific columns
round_data_values(xls_reduced, xls_rounded)

# Step 4: Apply data transformations (e.g., Box-Cox)
transform_data(xls_rounded, xls_transformed)

# Step 5: Filter outliers using Z-score thresholds
filter_outliers_by_zscore("Number of cycles (Box-Cox transformed)", xls_transformed, xls_filtered_bc, threshold=3)
filter_outliers_by_zscore("Number of cycles (Log transformed)", xls_transformed, xls_filtered_ln, threshold=3)

# Step 6: Split the filtered dataset into training, testing, and validation sets
split_data_into_train_test_validate(xls_filtered_bc, xls_divided)

# Load filtered dataset to create a mapping of measured cycles to authors
df = pd.read_excel(xls_filtered_bc)
cycle_to_author = dict(zip(df['Number of cycles (times)'], df['Author']))

# --------------------------------------------
# HYPERPARAMETER PREPARATION
# --------------------------------------------

# Load hyperparameters from an Excel file
hy = pd.read_excel('excel_spreadsheets/hyperparameters.xlsx', sheet_name=actual_opt)

# Create a hyperparameter grid from the loaded DataFrame
hyperparameter_grid = []
for index, row in hy.iterrows():
    hyperparams = {
        'num_layers': row['num_layers'],
        'dense': row['dense'],
        'optimizer': row['optimizer'],
        'activation_function': row['activation_function'],
        'n_epochs': row['n_epochs']
    }
    hyperparameter_grid.append(hyperparams)

# Print the generated hyperparameter grid
print(hyperparameter_grid)

# --------------------------------------------
# K-FOLD CROSS-VALIDATION SETUP
# --------------------------------------------

# Initialize the results DataFrame and list
results_df = pd.DataFrame(
    columns=['num_layers', 'dense', 'optimizer', 'activation_function', 'n_epochs',
             'min_train_loss', 'best_train_epoch', 'min_val_loss', 'best_val_epoch',
             'rmse_test', 'rmse_all', 'r2_test', 'r2_all']
)
results_list = []

# Load the filtered dataset
df = pd.read_excel(xls_filtered_bc)

# Define input and output columns
input_columns = ['Binder content (%)', 'Initial strain (µɛ)', 'Air Voids (%)']
output_column = 'Number of cycles (times)'

# Define the number of splits for K-Fold Cross-Validation
n_splits = 4
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# --------------------------------------------
# TRAINING AND EVALUATION LOOP
# --------------------------------------------

# Loop through each fold
results = []
all_predictions = []
kf_index = 0
for train_index, test_index in kf.split(df):
    kf_index += 1
    df_train, df_test = df.iloc[train_index].copy(), df.iloc[test_index].copy()
    df_train.reset_index(drop=True, inplace=True)

    # Save training and testing datasets for the current fold
    excel_train_test = file_name + f'_{kf_index}_train_test.xlsx'
    with pd.ExcelWriter(excel_train_test) as writer:
        df_train.to_excel(writer, sheet_name='Train', index=False)
        df_test.to_excel(writer, sheet_name='Test', index=False)

    # Scale the input features
    scaler = MinMaxScaler()
    train_indices, val_indices = train_test_split(np.arange(len(df_train)), test_size=0.2, random_state=42)

    # Prepare training, validation, and testing data
    train_inputs = scaler.fit_transform(df_train.loc[train_indices, input_columns])
    validate_inputs = scaler.transform(df_train.loc[val_indices, input_columns])
    test_inputs = scaler.transform(df_test[input_columns])

    train_outputs = df_train.loc[train_indices, output_column].values
    validate_outputs = df_train.loc[val_indices, output_column].values
    test_outputs = df_test[output_column].values

    iter_num = 1

    # Hyperparameter grid loop
    for params in hyperparameter_grid:
        # Extract hyperparameters
        num_layers = params['num_layers']
        dense = params['dense']
        optimizer = params['optimizer']
        activation_function = params['activation_function']
        n_epochs = int(params['n_epochs'])

        # Print iteration details
        print(f"Iteration: {iter_num}, K-Fold: {kf_index}, Layers: {num_layers}, Neurons: {dense}, Optimizer: "
              f"{optimizer}, Activation: {activation_function}, Epochs: {n_epochs}")
        iter_num += 1

        # Start timer for training duration
        start_time = time.time()
        print(time.strftime("%H:%M:%S"))

        # Train the model and evaluate its performance
        dir_name, train_predictions, test_predictions, validate_predictions, train_outputs, test_outputs, \
            validate_outputs, min_train_loss, best_train_epoch, min_val_loss, best_val_epoch, model = \
            train_ann_log_cv(
                kf_index=kf_index,
                actual_opt=actual_opt,
                actual_temp=actual_temp,
                train_inputs=train_inputs,
                test_inputs=test_inputs,
                validate_inputs=validate_inputs,
                train_outputs=train_outputs,
                test_outputs=test_outputs,
                validate_outputs=validate_outputs,
                num_layers=num_layers,
                n_epochs=n_epochs,
                dense=dense,
                optimizer=optimizer,
                activation_function=activation_function
            )

        # Save predictions and actual values in DataFrames
        df_train_results = pd.DataFrame({
            'Actual': train_outputs,
            'Predicted': train_predictions,
            'Author': [cycle_to_author.get(value, "Unknown") for value in train_outputs]
        })

        df_validate_results = pd.DataFrame({
            'Actual': validate_outputs,
            'Predicted': validate_predictions,
            'Author': [cycle_to_author.get(value, "Unknown") for value in validate_outputs]
        })

        df_test_results = pd.DataFrame({
            'Actual': test_outputs,
            'Predicted': test_predictions,
            'Author': [cycle_to_author.get(value, "Unknown") for value in test_outputs]
        })

        # Save these DataFrames to an Excel file for the current fold and hyperparameters
        results_file = xls_predicted
        with pd.ExcelWriter(results_file) as writer:
            df_train_results.to_excel(writer, sheet_name='Train', index=False)
            df_validate_results.to_excel(writer, sheet_name='Validation', index=False)
            df_test_results.to_excel(writer, sheet_name='Test', index=False)

        # --------------------------------------------
        # PERFORMANCE EVALUATION AND LOGGING
        # --------------------------------------------

        # Calculate metrics for test data
        rmse_test = np.sqrt(mean_squared_error(test_outputs, test_predictions))
        r2_test = r2_score(test_outputs, test_predictions)
        rmsle_test = rmsle_calculation(test_outputs, test_predictions)

        # Aggregate predictions and outputs for all data
        all_outputs = np.concatenate([train_outputs, test_outputs, validate_outputs])
        all_predictions = np.concatenate([train_predictions, test_predictions, validate_predictions])

        # Calculate metrics for the entire dataset
        rmse_all = np.sqrt(mean_squared_error(all_outputs, all_predictions))
        r2_all = r2_score(all_outputs, all_predictions)
        rmsle_all = rmsle_calculation(all_outputs, all_predictions)

        # Copy relevant files to the current directory
        for xls_file in [xls_separated, xls_rounded, xls_filtered_bc, xls_divided, xls_predicted]:
            shutil.copy(xls_file, os.path.join(dir_name, os.path.basename(xls_file)))

        # --------------------------------------------
        # LOGGING RESULTS FOR CURRENT CONFIGURATION
        # --------------------------------------------

        # Store the results for this hyperparameter set and fold in a dictionary
        results_dict = {
            'fold_num': kf_index,
            'loss_optimization': actual_opt,
            'dataset_temp': actual_temp,
            'num_layers': num_layers,
            'dense': dense,
            'optimizer': optimizer,
            'activation_function': activation_function,
            'n_epochs': n_epochs,
            'min_train_loss': min_train_loss,
            'best_train_epoch': best_train_epoch,
            'min_val_loss': min_val_loss,
            'best_val_epoch': best_val_epoch,
            'rmse_test': rmse_test,
            'rmsle_test': rmsle_test,
            'r2_test': r2_test,
            'rmse_all': rmse_all,
            'rmsle_all': rmsle_all,
            'r2_all': r2_all
        }

        results_list.append(results_dict)

        # Save aggregated results periodically to avoid data loss
        temp_df = pd.DataFrame(results_list)
        temp_df.to_excel('excel_spreadsheets/' + actual_opt + actual_temp + '_hyperparameter_tuning_results.xlsx',
                         index=False)

        # --------------------------------------------
        # LOGGING TRAINING DURATION
        # --------------------------------------------

        # Measure and log the training duration for the current hyperparameter set
        end_time = time.time()
        training_duration = end_time - start_time
        print(f"Training duration for set {iter_num}: {training_duration:.0f} seconds")

# Convert the results list to a DataFrame
results_df = pd.DataFrame(results_list)
