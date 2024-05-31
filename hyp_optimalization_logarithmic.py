import time
import shutil
import pandas as pd
from aux_functions.excel_modification import *
from aux_functions.train_ml_model import *
from aux_functions.print_automatic import *
import keras

# Defining options for datasets based on temperatures
twenty_deg = '20_deg'

# Defining options for loss optimization (either logarithmic or linear)
lin = 'lin'
log = 'log'

# Setting the currently used temperature and optimization type
actual_temp = twenty_deg
actual_opt = log

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
filter_outliers_by_quantile(xls_rounded, xls_filtered, 0.03, 0.90)  # Filters outliers using quantiles
filter_outliers_by_zscore(xls_filtered, xls_filtered, threshold=3)  # Filters outliers using Z-score method
split_data_into_train_test_validate(xls_filtered, xls_divided)  # Splits data into training, testing, and validation sets

# Loading the divided dataset
df_train_imputed = pd.read_excel(xls_divided, sheet_name='fatigue data - train')
df_test_imputed = pd.read_excel(xls_divided, sheet_name='fatigue data - test')
df_validate_imputed = pd.read_excel(xls_divided, sheet_name='fatigue data - validate')

# Reading hyperparameters from Excel file
hy = pd.read_excel('excel_spreadsheets/hyperparameters.xlsx', sheet_name=actual_opt)

# Preparing the hyperparameter grid by iterating over the hyperparameters DataFrame
hyperparameter_grid = []
for index, row in hy.iterrows():
    hyperparams = {'num_layers': row['num_layers'], 'dense': row['dense'], 'optimizer': row['optimizer'],
                   'activation_function': row['activation_function'], 'n_epochs': row['n_epochs']}
    hyperparameter_grid.append(hyperparams)

# Printing the prepared hyperparameter grid
print(hyperparameter_grid)

# Initialize a DataFrame to store results
results_df = pd.DataFrame(
    columns=['num_layers', 'dense', 'optimizer', 'activation_function', 'n_epochs', 'min_train_loss', 'best_train_epoch',
             'min_val_loss', 'best_val_epoch', 'rmse_test', 'rmse_all', 'r2_test', 'r2_all'])
results_list = []

# Read Excel file with predictions
df = pd.read_excel(xls_filtered)
predictions = []

# Define input and output columns
input_columns = ['Binder content (%)', 'Initial strain (µɛ)', 'Air Voids (%)']
output_column = 'Number of cycles (times)'

# Setting up K-Fold Cross-Validation
n_splits = 4
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Initialize an empty list to store results
results = []
all_predictions = []
kf_index = 0

# Iterating over each fold in the K-Fold Cross-Validation
for train_index, test_index in kf.split(df):
    kf_index += 1
    df_train, df_test = df.iloc[train_index].copy(), df.iloc[test_index].copy()
    df_train.reset_index(drop=True, inplace=True)

    # Preparing and saving training and testing data for each fold
    excel_train_test = file_name + f'_{kf_index}_train_test.xlsx'
    with pd.ExcelWriter(excel_train_test) as writer:
        df_train.to_excel(writer, sheet_name='Train', index=False)
        df_test.to_excel(writer, sheet_name='Test', index=False)

    # Feature scaling for input features
    scaler = MinMaxScaler()

    # Generate indices for splitting
    train_indices, val_indices = train_test_split(np.arange(len(df_train)), test_size=0.2, random_state=42)

    # Create training and validation subsets
    train_inputs = scaler.fit_transform(df_train.loc[train_indices, input_columns])
    validate_inputs = scaler.transform(df_train.loc[val_indices, input_columns])
    test_inputs = scaler.transform(df_test[input_columns])

    train_outputs = df_train.loc[train_indices, output_column].values
    validate_outputs = df_train.loc[val_indices, output_column].values
    test_outputs = df_test[output_column].values

    iter_num = 1

    # Iterating over each set of hyperparameters, training models, and evaluating them
    for params in hyperparameter_grid:
        num_layers = params['num_layers']
        dense = params['dense']
        optimizer = params['optimizer']
        activation_function = params['activation_function']
        n_epochs = int(params['n_epochs'])

        # Log the iteration and hyperparameter set being processed
        print(f"Iteration: {iter_num}, K-Fold: {kf_index}, Layers: {num_layers}, Neurons: {dense}, Optimizer: "
              f"{optimizer}, Activation: {activation_function}, Epochs: {n_epochs}")
        iter_num += 1

        # Start timer for training duration
        start_time = time.time()
        print(time.strftime("%H:%M:%S"))

        reset_random_seeds()  # Reset random seeds for reproducibility

        # Train the ANN model with the current set of hyperparameters and get the performance results
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

        # Save predictions in the DataFrame
        df_train.loc[train_indices, 'Training Predicted Cycles'] = train_predictions
        df_train.loc[val_indices, 'Validation Predicted Cycles'] = validate_predictions
        df_test.loc['Predicted Cycles'] = test_predictions

        with pd.ExcelWriter(excel_train_test, engine='openpyxl', mode='a') as writer:
            if 'Train_Full' in writer.book.sheetnames:
                del writer.book['Train_Full']
            df_train.to_excel(writer, sheet_name='Train_Full', index=False)

            if 'Test' in writer.book.sheetnames:
                del writer.book['Test']
            df_test.to_excel(writer, sheet_name='Test', index=False)

        shutil.copy(excel_train_test, os.path.join(dir_name, 'asphalt_20_deg_predicted.xlsx'))

        # Evaluate and log the model's performance
        ann_performance_report_separatly_dec(dir_name, train_outputs, train_predictions, test_outputs, test_predictions,
                                             validate_outputs, validate_predictions)
        rmse_test = np.sqrt(mean_squared_error(test_outputs, test_predictions))
        r2_test = r2_score(test_outputs, test_predictions)
        rmsle_test = rmsle_calculation(test_outputs, test_predictions)

        ann_performance_report_all_dec(dir_name, train_outputs, train_predictions, test_outputs, test_predictions,
                                       validate_outputs, validate_predictions)
        all_outputs = np.concatenate([train_outputs, test_outputs, validate_outputs])
        all_predictions = np.concatenate([train_predictions, test_predictions, validate_predictions])

        rmse_all = np.sqrt(mean_squared_error(all_outputs, all_predictions))
        r2_all = r2_score(all_outputs, all_predictions)
        rmsle_all = rmsle_calculation(all_outputs, all_predictions)

        # Copy relevant files to the current directory
        for xls_file in [xls_separated, xls_rounded, xls_filtered, xls_divided]:
            shutil.copy(xls_file, os.path.join(dir_name, os.path.basename(xls_file)))

        # Store results for each hyperparameter set and fold in a dictionary
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

        # Optionally, periodically save aggregated results to an Excel spreadsheet
        temp_df = pd.DataFrame(results_list)
        temp_df.to_excel('excel_spreadsheets/' + actual_opt + actual_temp + '_hyperparameter_tuning_results.xlsx',
                         index=False)

        # Calculate and log the training duration for the current hyperparameter set
        end_time = time.time()
        training_duration = end_time - start_time
        print(f"Training duration for set {iter_num}: {training_duration:.0f} seconds")

# Convert the results list to a DataFrame
results_df = pd.DataFrame(results_list)
