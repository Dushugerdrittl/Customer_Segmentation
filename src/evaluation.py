# src/evaluation.py
from sklearn.metrics import silhouette_score

def evaluate_clustering(data, cluster_column='Cluster'):
    """
    Evaluates the clustering performance using silhouette score.
    """
    # Calculate silhouette score
    score = silhouette_score(data.select_dtypes(include='number'), data[cluster_column])
    return score

def print_clustering_summary(data):
    """
    Prints a summary of the clusters, including their size and average spending.
    """
    print("Clustering Summary:")
    cluster_summary = data.groupby('Cluster').agg(
        customer_count=('Cluster', 'size'),
        avg_spending=('PURCHASES', 'mean')
    )
    print(cluster_summary)
