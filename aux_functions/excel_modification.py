import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from scipy.stats import boxcox


def separate_data(input_file: str, output_file: str) -> list:
    """
    Processes an Excel file to filter, and separate relevant data measured with 4PB testing method.

    This function performs the following steps:
    1. Reads the specified Excel file into a pandas DataFrame.
    2. Cleans column headers by removing additional spaces.
    3. Filters rows containing specific keywords ('4PB') in any column, ignoring case sensitivity.
    4. Specifies and retains only desired columns related to academic or scientific data (e.g., 'Author', 'DOI').
    5. Writes the processed DataFrame to a new Excel file without row indices.
    6. Returns a list of the retained column names.

    Args:
    - input_file (str): The path to the original Excel file.
    - output_file (str): The path to save the separated Excel file.

    Returns:
    - list: The list of columns retained in the separated Excel file.
    """

    # Read the Excel file
    df = pd.read_excel(input_file)

    # Remove additional spaces from column headers
    df.columns = df.columns.str.strip()

    # Filter rows based on specific keywords in any of the columns
    keywords = ['4PB']
    df = df[df.apply(lambda row: row.astype(str).str.contains('|'.join(keywords), case=False).any(), axis=1)]

    # List of columns to retain
    columns_to_keep = [
        'Author', 'DOI', 'Binder content (%)', 'Air Voids (%)', 'Initial strain (µɛ)', 'Number of cycles (times)'
    ]

    # Retain only the desired columns
    df = df[columns_to_keep]

    # Write the cleaned data to a new Excel file
    df.to_excel(output_file, index=False)

    return columns_to_keep


def drop_rows_with_missing_values(input_file: str, output_file: str, save: bool = True) -> pd.DataFrame:
    """
    Remove rows with any missing values from a dataset in an Excel file.

    The process involves:
    1. Loading the dataset from the specified Excel file into a pandas DataFrame. This step includes reading the data, which could be from a specified sheet if the sheet name is given.
    2. Dropping all rows from the DataFrame that contain at least one missing value. This operation ensures that the dataset used in subsequent analyses or models is free of any gaps that could introduce bias or errors.
    3. If the save flag is set to True, the cleaned DataFrame, now devoid of any rows with missing values, is saved to a new Excel file specified by the output_file parameter. This step is optional and allows the user to retain a physical copy of the cleaned dataset for future use or inspection.
    4. The function then returns the cleaned DataFrame, providing immediate access to the processed data for in-memory analysis or further processing steps.

    This functionality is particularly useful in scenarios where the integrity of the dataset is paramount, and any missing data could compromise the quality of the analysis or the performance of data-driven models.

    Args:
    - input_file (str): The path to the Excel file containing the dataset.
    - output_file (str): The path where the cleaned dataset may be saved, if saving is enabled.
    - save (bool, optional): Flag indicating whether the cleaned dataset should be saved to a new Excel file. Defaults to True.

    Returns:
    - pd.DataFrame: The DataFrame with rows containing missing values removed.
    """

    # Load the data from the specified Excel sheet
    df = pd.read_excel(input_file)

    # Remove rows with any missing values
    df_cleaned = df.dropna()

    # Optionally save the cleaned dataframe to a new Excel file
    if save:
        df_cleaned.to_excel(output_file, index=False)

    return df_cleaned


def round_data_values(input_file: str, output_file: str) -> None:
    """
    Rounds specific columns in an Excel dataset to desired precision.

    The function follows these steps:
    1. Reads an Excel file into a pandas DataFrame.
    2. Defines a dictionary specifying the columns to be rounded and their respective rounding precision.
       The precision is defined as the number of decimal places to round to. A positive number rounds to
       that many decimal places. A negative number rounds to the nearest 10, 100, 1000, etc., as specified
       by the negative precision.
    3. Iterates through the DataFrame, applying the specified rounding rules to each column listed in the dictionary.
    4. Saves the DataFrame with rounded values to a new Excel file specified by the output_file parameter, without including row indices.

    Args:
    - input_file (str): Path to the input Excel file.
    - output_file (str): Path where the rounded data will be saved.

    Returns:
    - None
    """

    # Load data from Excel file
    df = pd.read_excel(input_file)

    # Define columns and their respective rounding precision
    rounding_rules = {
        'Binder content (%)': 1,
        'Air Voids (%)': 1,
        'Initial strain (µɛ)': -1,
        'Number of cycles (times)': -1
    }

    # Apply rounding rules to each column
    for col, precision in rounding_rules.items():
        df[col] = df[col].round(precision)

    # Save the rounded data to the output Excel file
    df.to_excel(output_file, index=False)


def transform_data(input_file: str, output_file: str) -> None:
    """
    Applies a Box-Cox transformation and a natural logarithm transformation
    to the 'Number of cycles (times)' column in an Excel file.
    Adds the transformed values as new columns and saves the updated dataset.

    Args:
    - input_file (str): Path to the input Excel file.
    - output_file (str): Path where the transformed data will be saved.

    Returns:
    - None
    """
    # Load data from Excel file
    df = pd.read_excel(input_file)

    # Apply Box-Cox transformation
    transformed_values, lambda_value = boxcox(df['Number of cycles (times)'])

    # Apply natural logarithm transformation
    log_transformed_values = np.log(df['Number of cycles (times)'])

    # Add the transformed columns to the DataFrame
    df['Number of cycles (Box-Cox transformed)'] = transformed_values
    df['Number of cycles (Log transformed)'] = log_transformed_values

    # Save the updated DataFrame to a new Excel file
    df.to_excel(output_file, index=False)


def filter_outliers_by_quantile(input_file: str, output_file: str, lower_quantile: float = 0.05,
                                upper_quantile: float = 0.95) -> None:
    """
    Removes rows from an Excel dataset based on specified quantiles.

    Args:
    - input_file (str): Path to the input Excel file.
    - output_file (str): Path where the filtered data will be saved.
    - lower_quantile (float, optional): Lower quantile threshold. Default is 0.05 (5%).
    - upper_quantile (float, optional): Upper quantile threshold. Default is 0.95 (95%).

    Returns:
    - None
    """
    # Load data
    df = pd.read_excel(input_file)

    column = 'Number of cycles (times)'
    # Check if the specified column exists in the DataFrame
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in the input data")

    # Select the specified column
    col_data = df[column]

    # Calculate quantile thresholds
    lower_bound = col_data.quantile(lower_quantile)
    upper_bound = col_data.quantile(upper_quantile)

    # Create a mask to filter rows based on the quantile thresholds
    mask = (col_data >= lower_bound) & (col_data <= upper_bound)

    # Apply the mask to filter the DataFrame
    df_filtered = df[mask]

    # Save the filtered data
    df_filtered.to_excel(output_file, index=False)


def filter_outliers_by_zscore(filtrated_column, input_file: str, output_file: str, threshold: float = 3.0) -> None:
    """
    Removes rows from an Excel dataset based on Z-scores calculated only for specified columns.

    The function calculates Z-scores for the specified columns ("Box cox" and "LN") and filters out rows
    where the Z-score exceeds the specified threshold.

    Args:
    - filtrated_column: Box-cox transformed, or logarithmic transformed
    - input_file (str): Path to the input Excel file.
    - output_file (str): Path where the filtered data will be saved.
    - threshold (float, optional): Z-score threshold to identify outliers. Default is 3.0.

    Returns:
    - None
    """
    # Load data
    df = pd.read_excel(input_file)

    required_columns = [filtrated_column]

    # Calculate Z-scores only for the specified columns
    z_scores = stats.zscore(df[required_columns])

    # Handle NaN values in Z-scores
    z_scores = np.nan_to_num(z_scores)

    # Filter rows based on Z-score threshold
    mask = (np.abs(z_scores) < threshold).all(axis=1)
    df_filtered = df[mask]

    # Save the filtered data
    df_filtered.to_excel(output_file, index=False)


def add_predictions_to_table(input_file: str, output_file: str, train_preds, validation_preds,
                             reshape_method=None) -> None:
    """
    Adds predictions to train and test datasets in an Excel file.

    Functionality includes:
    1. Reading the specified Excel file to access different datasets without fully loading them into memory, thus optimizing performance.
    2. Associating the provided predicted values with their corresponding datasets based on predefined sheet names within the Excel file ('fatigue data - train' and 'fatigue data - test').
    3. Optionally reshaping the prediction arrays before appending them to the datasets, with supported operations for flattening nested lists or reshaping arrays to 1D format based on the specified `reshape_method`.
    4. Appending the predictions as a new column named 'Predicted' to the respective datasets.
    5. Saving the updated datasets, now inclusive of prediction values, back into an Excel file, utilizing the original sheet names for ease of identification and further analysis.

    This streamlined process allows for the direct integration of model predictions with the original datasets, facilitating an efficient workflow for model evaluation and analysis.

    Args:
    - input_file (str): The path to the input Excel file that contains the datasets.
    - output_file (str): The path where the datasets with added predictions will be saved.
    - train_preds (list or array): Predicted values for the training set.
    - test_preds (list or array): Predicted values for the test set.
    - reshape_method (str, optional): Method for reshaping the predictions if necessary ('flatten' or 'reshape').
                                      If None, predictions are appended as provided.

    Returns:
    - None
    """

    # Read the Excel file without loading data into memory
    xls = pd.ExcelFile(input_file)

    # Map each sheet name to its corresponding predicted values
    datasets = {
        'fatigue data - train': train_preds,
        'fatigue data - validate': validation_preds
    }

    # Create an Excel writer object to save data
    with pd.ExcelWriter(output_file) as writer:
        for sheet_name, predictions in datasets.items():
            # Load data for the current sheet into a DataFrame
            df = pd.read_excel(xls, sheet_name)

            # Reshape predictions if needed
            if reshape_method == 'flatten':
                predictions = [item for sublist in predictions for item in sublist]
            elif reshape_method == 'reshape':
                predictions = np.reshape(predictions, -1)

            # Add the reshaped predictions to the DataFrame
            df['Predicted'] = predictions

            # Save the modified dataframe to a sheet in the new Excel file
            df.to_excel(writer, sheet_name=sheet_name, index=False)

def split_data_into_train_test_validate(input_file: str, output_file: str) -> str:
    """
    Splits data from an Excel file into train, test, and validate datasets.

    The procedure is as follows:
    1. Loads the dataset from the specified Excel file into a pandas DataFrame.
    2. Splits the DataFrame into a training set and a temporary set (combination of test and validate)
       with a ratio of 63% for training and 37% for the temporary set.
    3. Further splits the temporary set into test and validate sets. The split is done evenly,
       but given the proportions from the initial split, this results in approximately 10% of
       the original data for testing and 27% for validation.
    4. Saves the three datasets to separate sheets in a single Excel file specified by the output_file parameter.
       The sheets are named 'fatigue data - train', 'fatigue data - validate', and 'fatigue data - test'.
    5. Returns the path to the output Excel file where the datasets were saved.

    Args:
    - input_file (str): Path to the input Excel file.
    - output_file (str): Path where the split datasets will be saved.

    Returns:
    - str: Path where the split datasets were saved.
    """

    # Load data from Excel file
    df = pd.read_excel(input_file)

    # Split data into train and a temporary set (test + validate)
    train, temp = train_test_split(df, test_size=0.37, random_state=28)

    # Further split the temporary set into test and validate sets
    test_ratio = 0.5
    test, validate = train_test_split(temp, test_size=test_ratio, random_state=39)

    # Save each dataset to a separate sheet in the output Excel file
    with pd.ExcelWriter(output_file) as writer:
        train.to_excel(writer, sheet_name='fatigue data - train', index=False)
        validate.to_excel(writer, sheet_name='fatigue data - validate', index=False)
        test.to_excel(writer, sheet_name='fatigue data - test', index=False)

    return output_file

def split_data_train_validate(input_file: str, output_file: str) -> str:
    """
    Splits data from an Excel file into train, test, and validate datasets.

    The procedure is as follows:
    1. Loads the dataset from the specified Excel file into a pandas DataFrame.
    2. Splits the DataFrame into a training set and a temporary set (combination of test and validate)
       with a ratio of 63% for training and 37% for the temporary set.
    3. Further splits the temporary set into test and validate sets. The split is done evenly,
       but given the proportions from the initial split, this results in approximately 10% of
       the original data for testing and 27% for validation.
    4. Saves the three datasets to separate sheets in a single Excel file specified by the output_file parameter.
       The sheets are named 'fatigue data - train', 'fatigue data - validate', and 'fatigue data - test'.
    5. Returns the path to the output Excel file where the datasets were saved.

    Args:
    - input_file (str): Path to the input Excel file.
    - output_file (str): Path where the split datasets will be saved.

    Returns:
    - str: Path where the split datasets were saved.
    """

    # Load data from Excel file
    df = pd.read_excel(input_file)

    # Split data into train and a temporary set (test + validate)
    train, validate = train_test_split(df, test_size=0.2, random_state=28)

    # Save each dataset to a separate sheet in the output Excel file
    with pd.ExcelWriter(output_file) as writer:
        train.to_excel(writer, sheet_name='fatigue data - train', index=False)
        validate.to_excel(writer, sheet_name='fatigue data - validate', index=False)

    return output_file

