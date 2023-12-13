import json
from scipy.spatial.distance import cdist

def find_k_nearest_neighbors(similarity_df, k):
    """
    Find k-nearest neighbors for each user.

    Args:
    similarity_df (DataFrame): DataFrame containing cosine similarity scores.
    k (int): Number of nearest neighbors to find.

    Returns:
    dict: A dictionary where keys are userIDs and values are lists of neighbor IDs.
    """
    neighbors = {}
    for user in similarity_df.index:
        # Sort the users based on similarity score
        sorted_users = similarity_df.loc[user].sort_values(ascending=False)
        top_k_users = sorted_users.iloc[1:k+1].index.tolist()
        neighbors[user] = top_k_users
    return neighbors