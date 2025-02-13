"""
This script performs K-means clustering on input data and assigns clusters to embeddings.
It utilizes the FAISS library for efficient clustering and supports GPU acceleration.

Command line arguments:
- input_file (str): Path to the input file containing embeddings.
- output_file (str): Path to the output file where cluster assignments will be saved.
- model_name (str): Name of the SentenceTransformer model to use for computing embeddings.
- ncentroids (int): Number of centroids for K-means clustering.
- niter (int): Number of iterations for K-means clustering.
- downsample_to (int, optional): Number of samples to downsample to before clustering.
- save_path (str, optional): Path to save the trained K-means centroids.

Output:
- A file containing the cluster assignments for the input embeddings.
"""



import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm
import numpy as np
import faiss
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer

##
## kmeans clustering
##

def train_kmeans_clustering(x, ncentroids=1024, niter=50, verbose=True, downsample_to=None, save_path=None):
    """
    Perform K-means clustering on the input data.

    Args:
        x (numpy.ndarray): Input data matrix of shape (n_samples, n_features).
        ncentroids (int): Number of centroids.
        niter (int): Number of iterations for training.
        verbose (bool): Whether to print verbose information.

    Returns:
        numpy.ndarray: Cluster labels for the input data.
    """
    d = x.shape[1]
    if downsample_to is not None:
        x = x[:downsample_to]
    kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose, gpu=torch.cuda.is_available())
    kmeans.train(x)
    if save_path is not None:
        with open(save_path, 'wb') as f:
            np.save(f, kmeans.centroids)
    return kmeans


def assign_kmeans_clusters(embs, save_path=None, kmeans=None):
    """
    Assigns K-means clusters to the given embeddings.

    Args:
        embs (numpy.ndarray): Embeddings to be clustered.
        save_path (str): Path to the saved centroids.

    Returns:
        numpy.ndarray: Cluster indices for the embeddings.
    """
    if kmeans is None:
        centroids = np.load(save_path)
        n, d = centroids.shape
        kmeans = faiss.Kmeans(d, n, verbose=True, gpu=True, niter=0, nredo=0)
        kmeans.train(embs, init_centroids=centroids) # this ensures that kmeans.index is created
        assert np.sum(kmeans.centroids - centroids) == 0, "centroids are not the same" # sanity check
    
    cluster_distances, cluster_indices = kmeans.assign(embs)
    return cluster_indices



if __name__ == "__main__":
    import  argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_file', type=str, required=True)
    parser.add_argument('--input_col_name', type=str, required=True)
    parser.add_argument('--trained_sbert_model_name', type=str, required=True)
    parser.add_argument('--output_cluster_file', type=str, required=True)
    parser.add_argument('--output_data_file', type=str, required=True)
    # parameters for kmeans clustering
    parser.add_argument('--ncentroids', type=int, default=1024)
    parser.add_argument('--niter', type=int, default=200)
    args = parser.parse_args()

    df = pd.read_csv(args.input_data_file)

    sbert_model = SentenceTransformer(args.trained_sbert_model_name)
    embeddings = sbert_model.encode(df[args.input_col_name].tolist(), show_progress_bar=True)

    kmeans = train_kmeans_clustering(embeddings, ncentroids=args.ncentroids, niter=args.niter, save_path=args.output_cluster_file)
    clusters = assign_kmeans_clusters(embeddings, kmeans=kmeans)

    df["cluster"] = clusters
    df.to_csv(args.output_data_file, index=False)

"""

python merge_labels.py \
    --input_data_file ../../data/v3_discourse_summaries/news-discourse/all_extracted_discourse.csv.gz \
    --input_col_name discourse_label \
    --trained_sbert_model_name models/mpnet-base-all-nli-triplet/trained-model \
    --output_cluster_file ../../data/v3_discourse_summaries/news-discourse/cluster_centroids.npy \
    --output_data_file ../../data/v3_discourse_summaries/news-discourse/all_extracted_discourse_with_clusters.csv
"""