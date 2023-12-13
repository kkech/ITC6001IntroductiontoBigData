import pandas as pd
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np


# Step 1: Load the Data
# Replace 'path_to_file' with the actual file paths
# user_artists_path = './hetrec2011-lastfm-2k/user_artists.dat'
# user_tags_path = './hetrec2011-lastfm-2k/user_taggedartists.dat'

# # Reading the data files
# user_artists_df = pd.read_csv(user_artists_path, sep='\t')
# user_tags_df = pd.read_csv(user_tags_path, sep='\t')

# # Step 2: Data Analysis and Visualization

# # Frequency plot of the listening frequency of artists by users
# artist_listen_counts = user_artists_df.groupby('artistID')['weight'].sum()
# plt.figure(figsize=(10, 6))
# artist_listen_counts.sort_values(ascending=False).plot(kind='bar')
# plt.title('Artists by Listening Frequency')
# plt.xlabel('Artist ID')
# plt.ylabel('Listening Frequency')
# plt.show()

# # Frequency of tags per user
# tag_frequency_per_user = user_tags_df.groupby('userID')['tagID'].count()
# plt.figure(figsize=(10, 6))
# tag_frequency_per_user.sort_values(ascending=False).head(50).plot(kind='bar')
# plt.title('Top 50 Users by Tag Frequency')
# plt.xlabel('User ID')
# plt.ylabel('Number of Tags')
# plt.show()

def detect_outliers_iqr(data):
    """
    Detect outliers using the Interquartile Range (IQR) method.

    Args:
    data (Pandas Series): Input data for which outliers need to be detected.

    Returns:
    Pandas Series: Boolean series where True indicates outliers.
    """
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    return (data < lower_bound) | (data > upper_bound)

def detect_outliers_zscore(data, threshold=3):
    """
    Detect outliers in data using the z-score method.

    Args:
    data (array-like): Input data for which outliers need to be detected.
    threshold (float): Z-score threshold for identifying outliers. Default is 3.

    Returns:
    array-like: Boolean mask where True indicates outliers.
    """
    z_scores = np.abs((data - np.mean(data)) / np.std(data))
    return z_scores > threshold

def plot_frequency(dataframe, x_column, y_column, title, xlabel, ylabel, top_n=None, show_x_labels=True):
    # If top_n is provided, filter the dataframe to keep only the top_n categories based on frequency
    if top_n:
        top_categories = dataframe[x_column].value_counts().index[:top_n]
        dataframe = dataframe[dataframe[x_column].isin(top_categories)]

    frequency = dataframe.groupby(x_column)[y_column].sum().sort_values(ascending=False)
    
    plt.figure(figsize=(20, 10))
    plt.yscale('log')
    frequency.plot(kind='bar')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if not show_x_labels:
        plt.xticks([])
    
    plt.tight_layout()
    plt.show()

def get_dataset_samples(directory, num_rows=5, exclude_files=['readme.txt']):
    """
    Get full dataset or samples from each file in the dataset directory.

    Args:
    directory (str): Path to the dataset directory.
    num_rows (int or None): Number of rows to include in the sample, or None for the full dataset.

    Returns:
    dict: A dictionary with file names as keys and data samples as values.
    """
    samples = {}
    dataset_files = os.listdir(directory)
    alternative_encodings = ['latin1', 'ISO-8859-1', 'cp1252']

    for file in dataset_files:
        if file.lower() in exclude_files:
            continue

        file_path = os.path.join(directory, file)
        try:
            if num_rows is None:
                df = pd.read_csv(file_path, sep='\t')
            else:
                df = pd.read_csv(file_path, sep='\t', nrows=num_rows)
            samples[file] = df
        except UnicodeDecodeError:
            for encoding in alternative_encodings:
                try:
                    if num_rows is None:
                        df = pd.read_csv(file_path, sep='\t', encoding=encoding)
                    else:
                        df = pd.read_csv(file_path, sep='\t', nrows=num_rows, encoding=encoding)
                    samples[file] = df
                    break
                except UnicodeDecodeError:
                    continue
        except Exception as e:
            print(f"Error reading {file}: {e}")
            samples[file] = None

    return samples

def save_outliers_to_file(outliers, filename):
    with open(filename, 'w') as file:
        for outlier in outliers:
            file.write(str(outlier) + '\n')

def convert_samples_to_latex(samples):
    """
    Convert data samples to LaTeX table format.

    Args:
    samples (dict): A dictionary with file names as keys and data samples (DataFrame) as values.

    Returns:
    dict: A dictionary with file names as keys and LaTeX table strings as values.
    """
    latex_samples = {}
    for file, df in samples.items():
        if df is not None:
            latex_samples[file] = df.to_latex(index=False)
        else:
            latex_samples[file] = "Error in data"
    
    return latex_samples

