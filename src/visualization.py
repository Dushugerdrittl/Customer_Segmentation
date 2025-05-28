# src/visualization.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_cluster_distribution(data_path, output_path):
    """
    Generates a bar plot showing the distribution of customers in each cluster.
    Saves the plot to the given output path.
    """
    # Load clustered data
    data = pd.read_csv(data_path)

    # Plot the cluster distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Cluster', data=data)
    plt.title('Cluster Distribution')
    plt.xlabel('Cluster')
    plt.ylabel('Count')

    # Save the plot as an image
    plt.savefig(f"{output_path}/cluster_distribution.png")
    plt.close()

def plot_spending_trends(data_path, output_path):
    """
    Generates a bar plot showing the average spending trends per cluster.
    Saves the plot to the given output path.
    """
    # Load clustered data
    data = pd.read_csv(data_path)

    # Group by cluster and calculate the average purchase value
    spending_data = data.groupby('Cluster')['PURCHASES'].mean()

    # Plot the spending trends
    plt.figure(figsize=(8, 6))
    spending_data.plot(kind='bar')
    plt.title('Average Spending by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Average Spending')

    # Save the plot as an image
    plt.savefig(f"{output_path}/spending_trends.png")
    plt.close()
