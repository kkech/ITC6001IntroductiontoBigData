import pandas as pd

def checkMissingValue(df):
    """
    Check for missing values in the given DataFrame.

    Args:
    df (DataFrame): A pandas DataFrame to check for missing values.

    Returns:
    int: Total number of missing values in the DataFrame.
    """
    return df.isnull().sum().sum()