from q1 import *
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
from q2 import *
import numpy as np
import matplotlib.pyplot as plt
from general import *

dataset_directory = './hetrec2011-lastfm-2k'

# General Preprocessing Starts #

print("---- Processing General Starts ----")

samples = get_dataset_samples(dataset_directory, num_rows=None)

# Load dataframes from samples
user_artists_df = samples.get('user_artists.dat', pd.DataFrame())
user_taggedartists_df = samples.get('user_taggedartists.dat', pd.DataFrame())
user_taggedartists_timestamps_df = samples.get('user_taggedartists-timestamps.dat', pd.DataFrame())
user_friends_df = samples.get('user_friends.dat', pd.DataFrame())
artists_df = samples.get('artists.dat', pd.DataFrame())
tags_df = samples.get('tags.dat', pd.DataFrame())

# Check for missing values
missing_values_user_artists = checkMissingValue(user_artists_df)
missing_values_user_taggedartists = checkMissingValue(user_taggedartists_df)
missing_values_user_taggedartists_timestamps = checkMissingValue(user_taggedartists_timestamps_df)
missing_values_user_friends = checkMissingValue(user_friends_df)
missing_values_artists = checkMissingValue(artists_df)
missing_values_tags = checkMissingValue(tags_df)

# Print the results
print(f"Missing Values in user_artists.dat: {missing_values_user_artists}")
print(f"Missing Values in user_taggedartists.dat: {missing_values_user_taggedartists}")
print(f"Missing Values in user_taggedartists-timestamps.dat: {missing_values_user_taggedartists_timestamps}")
print(f"Missing Values in user_friends.dat: {missing_values_user_friends}")
print(f"Missing Values in artists.dat: {missing_values_artists}")
print(f"Missing Values in tags.dat: {missing_values_tags}")

missing_artists = artists_df[artists_df.isnull().any(axis=1)]
missing_artist_ids = missing_artists['id']
print("IDs of Artists with Missing Values:")
print(missing_artist_ids)

print("---- Processing General Ends ----")

# General Preprocessing Ends #

print("---- Processing Q1 Starts ----")

samples = get_dataset_samples(dataset_directory)

# Display the samples
# for file, sample in samples.items():
#     print(f"\nSample from {file}:\n", sample)

latex_samples = convert_samples_to_latex(samples)

# Printing the samples in LaTeX tabular format
# for file, sample_latex in latex_samples.items():
#     print(f"\nLaTeX Table for {file}:\n")
#     print(sample_latex)

# Get Data Samples End #

# Q1 Plots #

samples = get_dataset_samples(dataset_directory, num_rows=None)

user_artists_df = samples.get('user_artists.dat', pd.DataFrame())
user_tags_df = samples.get('user_taggedartists.dat', pd.DataFrame())

# print("OK")

# Plotting listening frequency of artists
if not user_artists_df.empty:
    plot_frequency(user_artists_df, 'artistID', 'weight', 
                    'Listening Frequency of Artists by Users', 'Artist ID', 'Listening Frequency', None, show_x_labels=False)
    
    plot_frequency(user_artists_df, 'artistID', 'weight', 
                    'Listening Frequency of Artists by Users', 'Artist ID', 'Listening Frequency', 50, show_x_labels=True)


# Plotting frequency of tags per user
if not user_tags_df.empty:
    plot_frequency(user_tags_df, 'userID', 'tagID', 
                    'Frequency of Tags per User', 'User ID', 'Tag Frequency', None, show_x_labels=False)
    
    plot_frequency(user_tags_df, 'userID', 'tagID', 
                    'Frequency of Tags per User', 'User ID', 'Tag Frequency', 50, show_x_labels=True)
# Q1 Plots End #

# Q1 Outliers #
samples = get_dataset_samples(dataset_directory, num_rows=None)

user_artists_df = samples.get('user_artists.dat', pd.DataFrame())
user_taggedartists_df = samples.get('user_taggedartists.dat', pd.DataFrame())

# Aggregating listening counts per artist
artist_total_listening = user_artists_df.groupby('artistID')['weight'].sum()
outliers_artists = detect_outliers_zscore(artist_total_listening)

# Detect outliers for users based on total listening time
users_total_listening = user_artists_df.groupby('userID')['weight'].sum()
outliers_users = detect_outliers_zscore(users_total_listening)

tag_usage_counts = user_taggedartists_df['tagID'].value_counts()
# print(tag_usage_counts.head(10))
outliers_tags = detect_outliers_zscore(tag_usage_counts)

# Plotting #

# Creating scatter plots
plt.figure(figsize=(15, 10))

plt.subplot(3, 1, 1)
plt.scatter(artist_total_listening.index, artist_total_listening, color='blue', alpha=0.7)
plt.title('Artist Total Listening Time')
plt.xlabel('Artist ID')
plt.ylabel('Total Listening Time')

plt.subplot(3, 1, 2)
plt.scatter(users_total_listening.index, users_total_listening, color='green', alpha=0.7)
plt.title('User Total Listening Time')
plt.xlabel('User ID')
plt.ylabel('Total Listening Time')

plt.subplot(3, 1, 3)
plt.scatter(tag_usage_counts.index, tag_usage_counts, color='red', alpha=0.7)
plt.title('Tag Usage Count')
plt.xlabel('Tag ID')
plt.ylabel('Usage Count')

plt.tight_layout()
plt.show()

#Log results

log_artist_total_listening = np.log(artist_total_listening + 1)
log_users_total_listening = np.log(users_total_listening + 1)
log_tag_usage_counts = np.log(tag_usage_counts + 1)

plt.figure(figsize=(15, 10))

plt.subplot(3, 1, 1)
plt.scatter(artist_total_listening.index, log_artist_total_listening, color='blue', alpha=0.7)
plt.title('Artist Total Listening Time (Log Transformed Counts)')
plt.xlabel('Artist ID')
plt.ylabel('Log of Total Listening Time')

plt.subplot(3, 1, 2)
plt.scatter(users_total_listening.index, log_users_total_listening, color='green', alpha=0.7)
plt.title('User Total Listening Time (Log Transformed Counts)')
plt.xlabel('User ID')
plt.ylabel('Log of Total Listening Time')

plt.subplot(3, 1, 3)
plt.scatter(tag_usage_counts.index, log_tag_usage_counts, color='red', alpha=0.7)
plt.title('Tag Usage Count (Log Transformed Counts)')
plt.xlabel('Tag ID')
plt.ylabel('Log of Usage Count')

plt.tight_layout()
plt.show()

# Plotting Ends #

# Print detected outliers
print("Outliers in Artists' Listening Frequency:", artist_total_listening[outliers_artists].index)
print("Outliers in Users' Total Listening Time:", users_total_listening[outliers_users].index)
print("Outliers in Tags' Usage Frequency:", tag_usage_counts[outliers_tags].index)

# Print the number of outliers
print("Number of Outliers in Artists' Listening Frequency:", outliers_artists.sum())
print("Number of Outliers in Users' Total Listening Time:", outliers_users.sum())
print("Number of Outliers in Tags' Usage Frequency:", outliers_tags.sum())

# IQR Outliers #

# Apply IQR method
outliers_artists_iqr = detect_outliers_iqr(artist_total_listening)
outliers_users_iqr = detect_outliers_iqr(users_total_listening)
outliers_tags_iqr = detect_outliers_iqr(tag_usage_counts)

# Print detected outliers using IQR method
print("Outliers in Artists' Listening Frequency (artistID) using IQR:", artist_total_listening[outliers_artists_iqr].index)
print("Outliers in Users' Total Listening Time (userID) using IQR:", users_total_listening[outliers_users_iqr].index)
print("Outliers in Tags' Usage Frequency (tagID) using IQR:", tag_usage_counts[outliers_tags_iqr].index)

# Print the number of outliers detected using IQR
print("Number of Outliers in Artists' Listening Frequency using IQR:", outliers_artists_iqr.sum())
print("Number of Outliers in Users' Total Listening Time using IQR:", outliers_users_iqr.sum())
print("Number of Outliers in Tags' Usage Frequency using IQR:", outliers_tags_iqr.sum())

# IQR Ends #

# Detect and save Z-score outliers
save_outliers_to_file(artist_total_listening[outliers_artists].index, 'outliers_artists_zscore.txt')
save_outliers_to_file(users_total_listening[outliers_users].index, 'outliers_users_zscore.txt')
save_outliers_to_file(tag_usage_counts[outliers_tags].index, 'outliers_tags_zscore.txt')
print("Z-Score files saved")


# Print and save IQR outliers
save_outliers_to_file(artist_total_listening[outliers_artists_iqr].index, 'outliers_artists_iqr.txt')
save_outliers_to_file(users_total_listening[outliers_users_iqr].index, 'outliers_users_iqr.txt')
save_outliers_to_file(tag_usage_counts[outliers_tags_iqr].index, 'outliers_tags_iqr.txt')
print("IQR files saved")


print("---- Processing Q1 Ends ----")

# Q1 Outliers End #

# Q2 Starts #

print("---- Processing Q2 Starts ----")

# Q2.1 Starts #

samples = get_dataset_samples(dataset_directory, num_rows=None)
user_artists_df = samples.get('user_artists.dat', pd.DataFrame())

user_item_matrix = user_artists_df.pivot(index='userID', columns='artistID', values='weight').fillna(0)
cosine_sim = cosine_similarity(user_item_matrix)
cosine_sim_df = pd.DataFrame(cosine_sim, index=user_item_matrix.index, columns=user_item_matrix.index)

user_pairs_similarity = []
for user1, user2 in combinations(cosine_sim_df.index, 2):
    similarity_score = cosine_sim_df.loc[user1, user2]
    user_pairs_similarity.append([user1, user2, similarity_score])

user_pairs_similarity_df = pd.DataFrame(user_pairs_similarity, columns=['User1', 'User2', 'Similarity'])
user_pairs_similarity_df.to_csv('user-pairs-similarity.data', index=False)
print("File user-pairs-similarity.data saved")
# Q2.1 Ends #

# Q2.2 Starts #

# Find k-nearest neighbors for k=3 and k=10
neighbors_k3 = find_k_nearest_neighbors(cosine_sim_df, 3)
neighbors_k10 = find_k_nearest_neighbors(cosine_sim_df, 10)

combined_neighbors = {"k=3": neighbors_k3, "k=10": neighbors_k10}
with open('neighbors-k-users.data', 'w') as file:
    json.dump(combined_neighbors, file)

print("File neighbors-k-users.data saved")

# Q2.2 Ends #

print("---- Processing Q2 Ends ----")


# Q2 End #

# Q3 Starts #

print("---- Processing Q3 Starts ----")

samples = get_dataset_samples(dataset_directory, num_rows=None)
user_taggedartists_timestamps_df = samples.get('user_taggedartists-timestamps.dat', pd.DataFrame())

user_taggedartists_timestamps_df['timestamp'] = pd.to_datetime(user_taggedartists_timestamps_df['timestamp'], unit='ms')

user_taggedartists_timestamps_df['year'] = user_taggedartists_timestamps_df['timestamp'].dt.year
user_taggedartists_timestamps_df['month'] = user_taggedartists_timestamps_df['timestamp'].dt.month
user_taggedartists_timestamps_df['year_month'] = user_taggedartists_timestamps_df['timestamp'].dt.to_period('M')

user_taggedartists_timestamps_df['trimester'] = user_taggedartists_timestamps_df['timestamp'].dt.to_period('Q')

# Define a custom function to calculate semester from the month
def get_semester(month):
    return 'S1' if month <= 6 else 'S2'

user_taggedartists_timestamps_df['semester'] = user_taggedartists_timestamps_df['month'].apply(get_semester)

monthly_data = user_taggedartists_timestamps_df.groupby('year_month').nunique()
trimester_data = user_taggedartists_timestamps_df.groupby('trimester').nunique()

# For semester data, we need to group by both year and the custom semester
user_taggedartists_timestamps_df['year_semester'] = user_taggedartists_timestamps_df['year'].astype(str) + '_' + user_taggedartists_timestamps_df['semester']
semester_data = user_taggedartists_timestamps_df.groupby('year_semester').nunique()

# Plotting
plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
monthly_data['userID'].sort_index().plot(kind='bar', title='Monthly User Activity')
plt.ylabel('Number of Users')

plt.subplot(1, 3, 2)
trimester_data['userID'].sort_index().plot(kind='bar', title='Trimester User Activity')
plt.ylabel('Number of Users')

plt.subplot(1, 3, 3)
semester_data['userID'].sort_index().plot(kind='bar', title='Semester User Activity')
plt.ylabel('Number of Users')

plt.tight_layout()
plt.show()




###############

trimester_activity = user_taggedartists_timestamps_df.groupby('trimester').agg({
    'userID': pd.Series.nunique,
    'artistID': pd.Series.nunique,
    'tagID': pd.Series.nunique
}).rename(columns={'userID': 'Unique Users', 'artistID': 'Unique Artists', 'tagID': 'Unique Tags'})

artist_counts = user_taggedartists_timestamps_df.groupby(['trimester', 'artistID']).size().reset_index(name='counts')
tag_counts = user_taggedartists_timestamps_df.groupby(['trimester', 'tagID']).size().reset_index(name='counts')

# Now we'll sort within each trimester to find the top 5.
top_artists_per_trimester = artist_counts.groupby('trimester').apply(lambda x: x.nlargest(5, 'counts')).reset_index(drop=True)
top_tags_per_trimester = tag_counts.groupby('trimester').apply(lambda x: x.nlargest(5, 'counts')).reset_index(drop=True)

# The resulting DataFrame 'trimester_activity' has the counts of unique users, artists, and tags per trimester.
# 'top_artists_per_trimester' and 'top_tags_per_trimester' have the top 5 artists and tags per trimester respectively.

# Output the results
# print("Activity per Trimester:")
# print(trimester_activity)
# print("\nTop 5 Artists per Trimester:")
# print(top_artists_per_trimester)
# print("\nTop 5 Tags per Trimester:")
# print(top_tags_per_trimester)

# Save the aggregated activity data per trimester to a CSV file
trimester_activity.to_csv('trimester_activity.csv')


artists_df = samples.get('artists.dat', pd.DataFrame())
tags_df = samples.get('tags.dat', pd.DataFrame())

top_artists_with_names = top_artists_per_trimester.merge(artists_df, left_on='artistID', right_on='id', how='left')
top_tags_with_names = top_tags_per_trimester.merge(tags_df, left_on='tagID', right_on='tagID', how='left')

top_artists_with_names.to_csv('top_artists_with_names_per_trimester.csv')
top_tags_with_names.to_csv('top_tags_with_names_per_trimester.csv')

print("Files saved successfully with artist and tag names.")

print("---- Processing Q3 Ends ----")

# Q3 Ends #

# Q4 Starts #

# Q4.A Starts #

print("---- Processing Q4 Starts ----")

samples = get_dataset_samples(dataset_directory, num_rows=None)

user_artists_df = samples.get('user_artists.dat', pd.DataFrame())
user_friends_df = samples.get('user_friends.dat', pd.DataFrame())

artists_per_user = user_artists_df.groupby('userID')['artistID'].size().reset_index(name='num_artists')
friends_per_user = user_friends_df.groupby('userID').size().reset_index(name='num_friends')
merged_data = pd.merge(artists_per_user, friends_per_user, on='userID')

all_num_artists_50 = all(merged_data['num_artists'] == 50)

unique_num_artists = merged_data['num_artists'].unique()
num_artists_counts = merged_data['num_artists'].value_counts()

print(num_artists_counts)

# Plotting the distribution of num_artists
plt.figure(figsize=(10, 6))
num_artists_counts.plot(kind='bar', logy=True)
plt.title('Distribution of Number of Artists per User (Log Scale)')
plt.xlabel('Number of Artists')
plt.ylabel('Count of Users')
plt.show()

print("Is num_artists always 50:", all_num_artists_50)
print("Unique values in num_artists:", unique_num_artists)

# Calculate correlations
correlation = merged_data['num_artists'].corr(merged_data['num_friends'])
spearman_correlation = merged_data['num_artists'].corr(merged_data['num_friends'], method='spearman')
kendall_tau_correlation = merged_data['num_artists'].corr(merged_data['num_friends'], method='kendall')

# Output the correlation coefficient
print("Metrics for # Artists Per User - # Friends")
print("Pearson's Correlation coefficient:", correlation)
print("Spearman's Correlation coefficient:", spearman_correlation)
print("Kendall's Tau Correlation coefficient:", kendall_tau_correlation)


# Q4.A Ends #

# Q4.B Starts #

samples = get_dataset_samples(dataset_directory, num_rows=None)

user_artists_df = samples.get('user_artists.dat', pd.DataFrame())
user_friends_df = samples.get('user_friends.dat', pd.DataFrame())

# Aggregate total listening time per user
total_listening_per_user = user_artists_df.groupby('userID')['weight'].sum().reset_index(name='total_listening_time')

# Count friends per user
friends_count_per_user = user_friends_df.groupby('userID').size().reset_index(name='num_friends')

merged_data = pd.merge(total_listening_per_user, friends_count_per_user, on='userID')

# print(merged_data.head(10))

# Calculate correlation coefficients
pearson_corr = merged_data['total_listening_time'].corr(merged_data['num_friends'], method='pearson')
spearman_corr = merged_data['total_listening_time'].corr(merged_data['num_friends'], method='spearman')
kendall_tau_corr = merged_data['total_listening_time'].corr(merged_data['num_friends'], method='kendall')

# Print correlation coefficients
print("Metrics for Listening Time - # Friends")
print("Pearson's Correlation coefficient:", pearson_corr)
print("Spearman's Correlation coefficient:", spearman_corr)
print("Kendall's Tau Correlation coefficient:", kendall_tau_corr)

# Q4.B Ends #

print("---- Processing Q4 Ends ----")


# Q4 Ends #





