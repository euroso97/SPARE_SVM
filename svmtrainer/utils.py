# svmtrainer_project/svmtrainer/utils.py
# This file can be used for utility functions.
# For example, loading data from specific file formats,
# or more complex preprocessing steps.

import pandas as pd

def load_tabular_data(filepath, label_column_name):
    """
    Loads tabular data from a CSV file.

    Args:
        filepath (str): Path to the CSV file.
        label_column_name (str): Name of the column containing the labels.

    Returns:
        tuple: (features_df, labels_series)
               - features_df (pd.DataFrame): DataFrame of features.
               - labels_series (pd.Series): Series of labels.
    """
    try:
        data_df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {filepath} was not found.")
    except Exception as e:
        raise Exception(f"Error reading CSV file: {e}")

    if label_column_name not in data_df.columns:
        raise ValueError(f"Label column '{label_column_name}' not found in the CSV file.")

    labels_series = data_df[label_column_name]
    features_df = data_df.drop(columns=[label_column_name])

    return features_df, labels_series