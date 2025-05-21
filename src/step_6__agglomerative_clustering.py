#!/usr/bin/env python
# agglomerative_clustering.py - Focused on hierarchical clustering only

import sys
import pandas as pd
import os
import warnings
import numpy as np
from tqdm.auto import tqdm
import argparse
import networkx as nx
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from pathlib import Path

# Import tree utility functions
from utils_trees import label_hierarchical_tree, plot_graph, save_hierarchical_tree

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
warnings.simplefilter(action='ignore')


def get_linkage_matrix_from_model(model, n_samples):
    """
    Convert an AgglomerativeClustering model to a linkage matrix compatible with scipy functions.
    
    Parameters:
    -----------
    model : AgglomerativeClustering
        Fitted AgglomerativeClustering model
    n_samples : int
        Number of original samples
        
    Returns:
    --------
    numpy.ndarray
        Linkage matrix in scipy format
    """
    # Number of original samples (i.e., leaves in the tree)
    counts = np.zeros(model.children_.shape[0])
    
    # For each merge, record the size of the new cluster
    for i, merge in enumerate(model.children_):
        child_cluster_sizes = np.zeros(2)
        
        for j, child in enumerate(merge):
            # If the child is a leaf node
            if child < n_samples:
                child_cluster_sizes[j] = 1
            else:
                # If the child is a cluster, get its size from the counts array
                child_cluster_sizes[j] = counts[child - n_samples] + 1
                
        # Record the size of the merged cluster
        counts[i] = child_cluster_sizes.sum()
    
    # Create linkage matrix in the scipy format
    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)
    
    return linkage_matrix


def optimal_hierarchical_cuts_dp(embeddings, min_clusters=2, max_clusters=None, 
                               min_cluster_size=2, method='ward', metric='euclidean',
                               n_samples=None, verbose=True):
    """
    Optimized version using dynamic programming approach.
    """
    if n_samples is None:
        n_samples = embeddings.shape[0]
    
    if max_clusters is None:
        max_clusters = n_samples // 2
    
    # Step 1: Compute linkage matrix using AgglomerativeClustering
    model = AgglomerativeClustering(
        n_clusters=None,  # Don't predefine number of clusters
        distance_threshold=0,  # Cut tree at zero distance -> n_clusters=n_samples
        linkage=method,
        metric=metric,
        compute_distances=True  # Ensure distances are computed
    )
    model.fit(embeddings)
    
    # Convert to scipy linkage matrix
    linkage_matrix = get_linkage_matrix_from_model(model, n_samples)
    
    # The rest of the function remains the same
    # Step 2: Precompute all possible clusterings - O(n²)
    # Map: number of clusters -> (threshold, labels)
    cluster_map = {}
    
    # Sort distances from smallest to largest for this preprocessing step
    distances = sorted(linkage_matrix[:, 2])
    
    if verbose:
        print("Precomputing clusterings...")
        
    prev_n_clusters = -1
    for dist in distances:
        labels = fcluster(linkage_matrix, dist, criterion='distance')
        n_clusters = len(np.unique(labels))
        
        # Only store if this creates a new number of clusters and is within our range
        if (n_clusters != prev_n_clusters and 
            min_clusters <= n_clusters <= max_clusters):
            
            # Check minimum cluster size
            valid = True
            for label in np.unique(labels):
                if np.sum(labels == label) < min_cluster_size:
                    valid = False
                    break
                    
            if valid:
                cluster_map[n_clusters] = (dist, labels)
                prev_n_clusters = n_clusters
                
                if verbose and n_clusters % 5 == 0:
                    print(f"Found clustering with {n_clusters} clusters at distance {dist:.4f}")
    
    # Step 3: Compute silhouette scores only for the unique clusterings - O(n² * k)
    # where k is the number of unique clusterings (much less than n)
    scores = {}
    for n_clusters, (dist, labels) in cluster_map.items():
        scores[n_clusters] = silhouette_score(embeddings, labels)
        
        if verbose:
            print(f"Clusters: {n_clusters}, Distance: {dist:.4f}, Silhouette: {scores[n_clusters]:.4f}")
        
    # Initialize DP table
    # dp[i] = (max silhouette score achievable with i clusters, prev best cluster count)
    dp = {i: (-1, None) for i in range(min_clusters, max_clusters + 1)}
    
    # Base case: fill in scores we have
    for n_clusters in scores:
        dp[n_clusters] = (scores[n_clusters], None)
    
    # Fill the DP table
    cluster_counts = sorted(scores.keys())
    
    # Reconstruct optimal sequence
    optimal_cluster_counts = []
    current = max(cluster_counts, key=lambda x: scores[x])
    
    while current is not None:
        optimal_cluster_counts.append(current)
        # Find next larger cluster count with best score
        next_clusters = [c for c in cluster_counts if c > current]
        if not next_clusters:
            break
        current = max(next_clusters, key=lambda x: scores[x])
    
    # Reverse to get from coarse to fine
    optimal_cluster_counts.reverse()
    
    # Get corresponding thresholds
    optimal_thresholds = [cluster_map[c][0] for c in optimal_cluster_counts]
    optimal_scores = [scores[c] for c in optimal_cluster_counts]
    
    # Create the hierarchical tree
    tree, cluster_node_ids = create_hierarchical_tree(embeddings, linkage_matrix, n_samples, optimal_thresholds, model)
    
    # Visualization (same as original function)
    if verbose:
        # [visualization code as in the original function]
        pass
    
    return {
        'optimal_thresholds': optimal_thresholds,
        'silhouette_scores': optimal_scores,
        'n_clusters': optimal_cluster_counts,
        'linkage_matrix': linkage_matrix,
        'tree': tree,
        'cluster_node_ids': cluster_node_ids
    }


def optimal_hierarchical_cuts(
    embeddings, 
    min_clusters=2,
    max_clusters=None, 
    min_cluster_size=2, 
    method='ward', 
    metric='euclidean',
    n_samples=None, 
    verbose=True
):
    """
    Find optimal hierarchical clustering cut distances using silhouette analysis.
    
    Parameters:
    -----------
    embeddings : numpy.ndarray
        The data matrix, shape (n_samples, n_features)
    min_clusters : int, default=2
        Minimum number of clusters to consider
    max_clusters : int or None, default=None
        Maximum number of clusters to consider. If None, uses n_samples/2
    min_cluster_size : int, default=2
        Minimum size a cluster must have to be considered valid
    method : str, default='ward'
        Linkage method for hierarchical clustering
    metric : str, default='euclidean'
        Distance metric for clustering
    n_samples : int or None, default=None
        Number of samples in the data. If None, uses embeddings.shape[0]
    verbose : bool, default=True
        Whether to print progress and plot results
        
    Returns:
    --------
    dict
        Contains:
        - 'optimal_thresholds': list of optimal distance thresholds
        - 'silhouette_scores': silhouette scores at each level
        - 'n_clusters': number of clusters at each level
        - 'linkage_matrix': the hierarchical clustering linkage matrix
        - 'tree': a networkx DiGraph representing the hierarchical tree
    """
    if n_samples is None:
        n_samples = embeddings.shape[0]
    
    if max_clusters is None:
        max_clusters = n_samples // 2
    
    model = AgglomerativeClustering(
        n_clusters=None,       # Don't predefine number of clusters
        distance_threshold=0,  # Cut tree at zero distance -> n_clusters=n_samples
        linkage=method, 
        metric=metric,
        compute_distances=True  # Ensure distances are computed
    )
    model.fit(embeddings)
    
    # Convert to scipy linkage matrix format
    linkage_matrix = get_linkage_matrix_from_model(model, n_samples)
    
    # Extract all possible threshold distances from the linkage matrix
    distances = sorted(linkage_matrix[:, 2], reverse=True)
    
    # Initialize variables to track results
    all_scores = []
    all_distances = []
    all_n_clusters = []
    
    if verbose:
        print("Evaluating silhouette scores across distances...")
    
    # Evaluate silhouette score for each possible threshold
    for i, dist in enumerate(distances):
        labels = fcluster(linkage_matrix, dist, criterion='distance')
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        # Skip if we have too few or too many clusters
        if n_clusters < min_clusters or n_clusters > max_clusters:
            continue
        
        # Check if all clusters meet minimum size requirement
        valid_clustering = True
        for label in unique_labels:
            if np.sum(labels == label) < min_cluster_size:
                valid_clustering = False
                break
                
        if not valid_clustering:
            continue
            
        # Calculate silhouette score
        try:
            score = silhouette_score(embeddings, labels)
            all_scores.append(score)
            all_distances.append(dist)
            all_n_clusters.append(n_clusters)
            
            if verbose and i % 10 == 0:
                print(f"Distance {dist:.4f}, Clusters: {n_clusters}, Silhouette: {score:.4f}")
        except:
            # This can happen if we have only one cluster or each point is its own cluster
            continue
    
    if len(all_scores) == 0:
        raise ValueError("No valid clusterings found. Try adjusting parameters.")
        
    # Convert to numpy arrays for easier manipulation
    all_scores = np.array(all_scores)
    all_distances = np.array(all_distances)
    all_n_clusters = np.array(all_n_clusters)
    
    # Find optimal thresholds using a hierarchical approach
    optimal_thresholds = []
    remaining_indices = np.arange(len(all_scores))
    
    while len(remaining_indices) > 0:
        # Find the current global maximum silhouette score
        max_idx = remaining_indices[np.argmax(all_scores[remaining_indices])]
        max_dist = all_distances[max_idx]
        max_clusters = all_n_clusters[max_idx]
        
        # Add this threshold to our optimal set
        optimal_thresholds.append(max_dist)
        
        # Remove all thresholds that have fewer clusters than the current one
        remaining_indices = np.array([i for i in remaining_indices if all_n_clusters[i] > max_clusters])
    
    optimal_thresholds = sorted(optimal_thresholds, reverse=True)
    tree, cluster_node_ids = create_hierarchical_tree(embeddings, linkage_matrix, n_samples, optimal_thresholds, model)
    optimal_scores = []
    optimal_n_clusters = []
    for threshold in optimal_thresholds:
        labels = fcluster(linkage_matrix, threshold, criterion='distance')
        score = silhouette_score(embeddings, labels)
        optimal_scores.append(score)
        optimal_n_clusters.append(len(np.unique(labels)))
    
    if verbose:
        print("\nOptimal distance thresholds:")
        for i, (dist, score, n_clusters) in enumerate(zip(optimal_thresholds, optimal_scores, optimal_n_clusters)):
            print(f"Level {i+1}: Distance={dist:.4f}, Clusters={n_clusters}, Silhouette={score:.4f}")
    
    return {
        'optimal_thresholds': optimal_thresholds,
        'silhouette_scores': optimal_scores,
        'n_clusters': optimal_n_clusters,
        'linkage_matrix': linkage_matrix,
        'tree': tree,
        'cluster_node_ids': cluster_node_ids,
        'model': model  # Include the model for reference
    }


def create_hierarchical_tree(embeddings, linkage_matrix=None, n_samples=None, distance_thresholds=None, model=None):
    """
    Create a hierarchical tree structure with multiple levels based on optimal distance thresholds.
    
    Parameters:
    -----------
    embeddings : numpy.ndarray
        The data matrix, shape (n_samples, n_features)
    linkage_matrix : numpy.ndarray, optional
        The linkage matrix from scipy's hierarchical clustering
    n_samples : int, optional
        The number of original data points
    distance_thresholds : list, optional
        A sorted list of distance thresholds (from largest to smallest)
    model : AgglomerativeClustering, optional
        Fitted AgglomerativeClustering model to use for more accurate tree construction
        
    Returns:
    --------
    networkx.DiGraph
        A directed graph representing the sparse hierarchical structure
    """
    if n_samples is None:
        n_samples = embeddings.shape[0]
    if distance_thresholds is None:
        distance_thresholds = sorted(linkage_matrix[:, 2], reverse=True)[:3]  # Default to top 3 levels
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add original data points as leaf nodes
    for i in range(n_samples):
        G.add_node(i, type='sample', level='leaf', orig_node_id=i)
    
    # Generate clusters at each threshold
    all_clusters = {}
    for threshold in distance_thresholds:
        cluster_labels = fcluster(linkage_matrix, threshold, criterion='distance')
        
        # Group samples by cluster
        clusters = {}
        for i in range(n_samples):
            cluster_id = int(cluster_labels[i])
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(i)
        
        all_clusters[threshold] = clusters
    
    # Create a unique ID for each cluster
    cluster_id_counter = n_samples
    cluster_node_ids = {}  # Maps (threshold, cluster_id) to node_id
    
    # Build the tree from bottom up (smallest threshold first)
    for threshold in reversed(distance_thresholds):
        clusters = all_clusters[threshold]
        
        for cluster_id, samples in clusters.items():
            # Create a unique node ID for this cluster
            node_id = cluster_id_counter
            cluster_id_counter += 1
            
            # Store the mapping
            cluster_node_ids[(threshold, cluster_id)] = node_id
            
            # Add the cluster node
            G.add_node(node_id, 
                      type='cluster',
                      distance_threshold=threshold,
                      size=len(samples),
                      samples=samples)
            
            # If this is the smallest threshold, connect to samples directly
            if threshold == distance_thresholds[-1]:
                for sample in samples:
                    G.add_edge(node_id, sample)
    
    # Now connect clusters across thresholds
    for i in range(len(distance_thresholds) - 1):
        higher_threshold = distance_thresholds[i]
        lower_threshold = distance_thresholds[i + 1]
        
        higher_clusters = all_clusters[higher_threshold]
        lower_clusters = all_clusters[lower_threshold]
        
        # For each higher threshold cluster, find its children in the lower threshold
        for higher_cluster_id, higher_samples in higher_clusters.items():
            higher_node_id = cluster_node_ids[(higher_threshold, higher_cluster_id)]
            
            # Find all lower clusters that contain samples from this higher cluster
            children = set()
            for lower_cluster_id, lower_samples in lower_clusters.items():
                # Use set intersection to check for overlap more efficiently
                if set(higher_samples).intersection(set(lower_samples)):
                    lower_node_id = cluster_node_ids[(lower_threshold, lower_cluster_id)]
                    children.add(lower_node_id)
            
            # Connect the higher cluster to its children
            for child_id in children:
                G.add_edge(higher_node_id, child_id)
    
    # Verify that the tree is properly connected
    if nx.number_weakly_connected_components(G) > 1:
        print("Warning: Tree has disconnected components. Adding a super-root to connect them.")
        components = list(nx.weakly_connected_components(G))
        super_root_id = cluster_id_counter
        
        # Add a super-root
        G.add_node(super_root_id, type='super_root', size=n_samples)
        
        # Connect each component to the super-root
        for component in components:
            # Find nodes with no incoming edges in this component
            for node in component:
                if G.in_degree(node) == 0 and node < n_samples:
                    G.add_edge(super_root_id, node)
                elif G.in_degree(node) == 0:
                    G.add_edge(super_root_id, node)
    
    return G, cluster_node_ids  

def run_hierarchical_clustering(
    centroids_file_path,
    output_dir_path,
    min_clusters=2,
    max_clusters=None,
    min_cluster_size=2,
    method='ward',
    metric='euclidean',
):
    """
    Perform hierarchical clustering on pre-computed centroids (from KMeans)
    and find optimal clustering levels.
    
    Parameters:
    -----------
    centroids_file_path : str
        Path to the NumPy file containing the centroids from KMeans clustering
    output_dir_path : str
        Directory to save the output files
    min_clusters : int, default=2
        Minimum number of clusters to consider
    max_clusters : int, default=None
        Maximum number of clusters to consider
    min_cluster_size : int, default=2
        Minimum size a cluster must have to be considered valid
    method : str, default='ward'
        Linkage method for hierarchical clustering
    metric : str, default='euclidean'
        Distance metric for clustering
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir_path, exist_ok=True)
    
    # Load centroids file
    print(f"Loading centroids from: {centroids_file_path}")
    try:
        centroids = np.load(centroids_file_path)
    except Exception as e:
        raise ValueError(f"Error loading centroids file: {e}")
    
    n_centroids = len(centroids)
    print(f"Loaded {n_centroids} centroids with dimension {centroids.shape[1]}")
    
    # Run hierarchical clustering
    print("Performing hierarchical clustering and finding optimal thresholds...")
    result = optimal_hierarchical_cuts(
        centroids,
        min_clusters=min_clusters,
        max_clusters=max_clusters,
        min_cluster_size=min_cluster_size,
        method=method,
        metric=metric,
        n_samples=n_centroids,
        verbose=True
    )
    
    # Save results
    output_base = Path(output_dir_path)
    
    # Save thresholds and scores
    thresholds_df = pd.DataFrame({
        'threshold': result['optimal_thresholds'],
        'n_clusters': result['n_clusters'],
        'silhouette_score': result['silhouette_scores']
    })
    thresholds_df.to_csv(output_base / 'optimal_thresholds.csv', index=False)
    print(f"Saved optimal thresholds to: {output_base / 'optimal_thresholds.csv'}")
    
    # Save hierarchical tree
    tree_path = output_base / 'hierarchical_tree.gml'
    save_hierarchical_tree(result['tree'], tree_path)
    print(f"Saved hierarchical tree to: {tree_path}")
    
    # For each threshold, save cluster assignments
    linkage_m = result['linkage_matrix']
    # This map is {(threshold, fcluster_label): graph_node_id}
    # fcluster_label is the 1,2,3... label from fcluster for a *cluster*
    # graph_node_id is the actual node ID in the NetworkX tree for that cluster
    threshold_to_fcluster_to_graph_node_map = result['cluster_node_ids'] 

    for i, threshold in enumerate(result['optimal_thresholds']):
        n_clusters_at_level = result['n_clusters'][i]
        
        # Get fcluster labels for each original sample (centroid) at this threshold
        # These labels are 1, 2, ..., n_clusters_at_level
        fcluster_labels_for_samples = fcluster(linkage_m, threshold, criterion='distance')
        
        # Map each sample's fcluster label to its corresponding graph_node_id
        graph_node_ids_for_samples = []
        for f_label in fcluster_labels_for_samples:
            # (threshold, f_label) is the key to get the graph_node_id for the cluster this sample belongs to
            graph_node = threshold_to_fcluster_to_graph_node_map.get((threshold, f_label))
            if graph_node is None:
                # This might happen if a threshold in optimal_thresholds wasn't used to build the cluster_node_ids map
                # or if an fcluster_label appears that wasn't in the map for that threshold.
                # This implies that the set of `distance_thresholds` passed to `create_hierarchical_tree`
                # (which is `optimal_thresholds`) must be comprehensive.
                print(f"Warning: Could not map fcluster label {f_label} at threshold {threshold} to a graph node ID. Appending None.")
                graph_node_ids_for_samples.append(None) 
            else:
                graph_node_ids_for_samples.append(int(graph_node))
        
        # Save cluster assignments
        cluster_df = pd.DataFrame({
            'centroid_id': np.arange(n_centroids), # K-Means centroid ID (0-indexed)
            'fcluster_label': fcluster_labels_for_samples, # fcluster output (1-indexed usually)
            'graph_node_id': graph_node_ids_for_samples # Actual tree node ID for the cluster
        })
        cluster_path = output_base / f'clusters_level_{i+1}_{n_clusters_at_level}_clusters.csv'
        cluster_df.to_csv(cluster_path, index=False)
        print(f"Saved level {i+1} clustering ({n_clusters_at_level} clusters) to: {cluster_path}")
        
    print("Hierarchical clustering completed successfully.")
    return result


def label_and_visualize_tree(
    tree,
    initial_cluster_labels_path,
    output_dir_path,
    leaf_node_counts=None,
    min_descendants=2,
    max_depth=8,
    num_samples_to_label=10,
    visualize=True,
    examples_metadata=None,
    use_discretization=False,
    discretization_goal="determine the common theme or purpose",
    initial_labels_embeddings_path=None
):
    """
    Label and visualize a hierarchical tree.
    
    Parameters:
    -----------
    tree : networkx.DiGraph
        The hierarchical tree to label and visualize
    initial_cluster_labels_path : str
        Path to the CSV file containing initial cluster labels
    output_dir_path : str
        Directory to save the labeled tree and visualization
    examples_dataframe_path : str, optional
        Path to the CSV file containing example sentences for each cluster
    leaf_node_counts : dict, optional
        Dictionary mapping node IDs to their respective counts
    min_descendants : int, default=2
        Minimum number of descendants required for a node to be retained
    max_depth : int, default=8
        Maximum depth of the tree to consider during pruning
    num_samples_to_label : int, default=10
        Number of samples to use for labeling each inner node
    visualize : bool, default=True
        Whether to generate and save a visualization of the labeled tree
    examples_metadata : dictionary
        contains keys: examples_dataframe_path, num_examples_per_node, cluster_col, sentence_col
        * examples_dataframe_path: path to the examples dataframe
        * num_examples_per_node: Number of example datapoints to use for each node (in addition to the labels, to keep it more specific.)
        * cluster_col: column name of the cluster column in the examples dataframe
        * sentence_col: column name of the sentence column in the examples dataframe
    use_discretization : bool, default=False
        Whether to use discretization for labeling
    discretization_goal : str, default="determine the common theme or purpose"
        The goal or purpose for discretization
    initial_labels_embeddings_path : str, optional
        Path to the initial labels embeddings file (required if use_discretization is True)
        
    Returns:
    --------
    tuple
        (labeled_tree, pruned_tree, inner_node_label_dict)
    """
    # Save labeled tree
    output_base = Path(output_dir_path)
    labeled_tree_path = output_base / 'labeled_hierarchical_tree.gml'
    pruned_tree_path = output_base / 'pruned_hierarchical_tree.gml'
    inner_node_labels_path = output_base / 'inner_node_labels.csv'
    
    # Load initial cluster labels
    print(f"Loading initial cluster labels from: {initial_cluster_labels_path}")
    initial_labels_df = pd.read_csv(initial_cluster_labels_path)
    
    # Ensure the labels dataframe has node_id and label columns
    if 'node_id' not in initial_labels_df.columns or 'label' not in initial_labels_df.columns:
        raise ValueError("Initial labels dataframe must have 'node_id' and 'label' columns")
    
    # Rename and process the dataframe
    initial_labels_df = initial_labels_df.rename(columns={'node_id': 'orig_node_id'})
    
    # Load examples dataframe if provided
    examples_dataframe = None
    if examples_metadata is not None:
        print(f"Loading examples from: {examples_metadata['examples_dataframe_path']}")
        examples_dataframe = pd.read_csv(examples_metadata['examples_dataframe_path'])
        if examples_metadata['cluster_col'] not in examples_dataframe.columns:
            print("Warning: Examples dataframe does not have a 'cluster' column")
            examples_dataframe = None
        elif examples_metadata['sentence_col'] not in examples_dataframe.columns:
            print("Warning: Examples dataframe does not have a 'sentence' column")
            examples_dataframe = None
        else:
            print(f"Loaded {len(examples_dataframe)} examples")
    
    # Add labels to leaf nodes in the tree
    for _, row in initial_labels_df.iterrows():
        node_id = int(row['orig_node_id'])
        for n in tree.nodes():
            if tree.nodes[n].get('orig_node_id') == node_id:
                nx.set_node_attributes(tree, {n: row['label']}, 'label')
                break
    
    # Generate leaf node counts if not provided
    if leaf_node_counts is None:
        leaf_node_counts = {}
        for node in tree.nodes():
            if tree.out_degree(node) == 0:  # Leaf node
                leaf_node_counts[node] = 1
    
    # Add size attribute to all nodes if missing. This is needed by label_hierarchical_tree to sample nodes by size
    for node in tree.nodes():
        if 'size' not in tree.nodes[node]:
            if 'samples' in tree.nodes[node]:
                nx.set_node_attributes(tree, {node: len(tree.nodes[node]['samples'])}, 'size')
            else:
                nx.set_node_attributes(tree, {node: 1}, 'size')
    
    # Load discretization resources if needed
    text_embeddings = None
    combined_cluster_texts = None
    if use_discretization:
        if initial_labels_embeddings_path is None:
            raise ValueError("initial_labels_embeddings_path must be provided when use_discretization is True")
        
        # Load the embeddings
        print(f"Loading embeddings from: {initial_labels_embeddings_path}")
        text_embeddings = np.load(initial_labels_embeddings_path)
        
        # Create combined texts from initial labels
        combined_cluster_texts = initial_labels_df.apply(
            lambda x: f"\"{x['label']}\": {x.get('description', '')}", axis=1
        ).tolist()
        
        print(f"Loaded {len(combined_cluster_texts)} texts and {text_embeddings.shape} embeddings for discretization")
    
    # Call the label_hierarchical_tree function
    print("Labeling hierarchical tree...")
    labeled_tree, pruned_tree, inner_node_label_dict = label_hierarchical_tree(
        tree,
        leaf_node_counts=leaf_node_counts,
        min_descendants=min_descendants,
        max_depth=max_depth,
        num_samples_to_label=num_samples_to_label,
        examples_dataframe=examples_dataframe,
        examples_metadata=examples_metadata,
        use_discretization=use_discretization,
        text_embeddings=text_embeddings,
        combined_cluster_texts=combined_cluster_texts,
        discretization_goal=discretization_goal
    )
    
    save_hierarchical_tree(labeled_tree, labeled_tree_path)
    print(f"Saved labeled tree to: {labeled_tree_path}")
    save_hierarchical_tree(pruned_tree, pruned_tree_path)
    print(f"Saved pruned tree to: {pruned_tree_path}")
    
    # Save inner node labels
    inner_node_label_df = pd.DataFrame([
        {'node_id': node_id, 'label': label_info['label'], 'description': label_info['description']}
        for node_id, label_info in inner_node_label_dict.items()
    ])
    inner_node_label_df.to_csv(inner_node_labels_path, index=False)
    print(f"Saved inner node labels to: {inner_node_labels_path}")
    
    # Visualize the labeled tree if requested
    if visualize:
        # Set up visualization parameters
        plot_path = output_base / 'labeled_tree_visualization.png'
        plt.figure(figsize=(16, 12))
        
        # Use plot_graph function to visualize
        plot_graph(
            pruned_tree,
            with_labels=True,
            with_node_id=True,
            with_sizes=True,
            label_name='label',
            node_color='lightblue',
            node_max_size=50,
            node_min_size=5,
            font_size=10
        )
        
        # Save the plot
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved tree visualization to: {plot_path}")
    
    print("Tree labeling and visualization completed successfully.")
    return labeled_tree, pruned_tree, inner_node_label_dict


def main():
    parser = argparse.ArgumentParser(description="Perform hierarchical clustering on KMeans centroids.")
    parser.add_argument("centroids_file", help="Path to the file containing KMeans centroids (NumPy format)")
    parser.add_argument("output_dir", help="Directory to save the output files")
    parser.add_argument("--min_clusters", type=int, default=2, help="Minimum number of clusters")
    parser.add_argument("--max_clusters", type=int, default=None, help="Maximum number of clusters")
    parser.add_argument("--min_cluster_size", type=int, default=2, help="Minimum size a cluster must have")
    parser.add_argument("--method", default="ward", choices=["ward", "complete", "average", "single"], 
                        help="Linkage method for hierarchical clustering")
    parser.add_argument("--metric", default="euclidean", help="Distance metric for clustering")
    
    # Add arguments for labeling and visualization
    parser.add_argument("--label_tree", action="store_true", help="Label the hierarchical tree using utils_trees functions")
    parser.add_argument("--initial_labels", help="Path to the CSV file containing initial cluster labels")
    parser.add_argument("--min_descendants", type=int, default=2, help="Minimum descendants for tree pruning")
    parser.add_argument("--max_depth", type=int, default=8, help="Maximum depth for tree pruning")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to use for labeling each inner node")
    parser.add_argument("--no_visualize", action="store_true", help="Skip tree visualization")

    # Add arguments for examples
    parser.add_argument("--examples_df", help="Path to the CSV file containing example sentences for each cluster")
    parser.add_argument("--num_examples_per_node", type=int, default=3, help="Number of example datapoints to use for each node (in addition to the labels, to keep it more specific.)")
    parser.add_argument("--examples_cluster_col", type=str, default="cluster", help="Column name of the cluster column in the examples dataframe")
    parser.add_argument("--examples_sentence_col", type=str, default="sentence", help="Column name of the sentence column in the examples dataframe")

    # Add arguments for discretization
    parser.add_argument("--use_discretization", action="store_true", help="Use discretization approach for labeling")
    parser.add_argument("--discretization_goal", type=str, default="determine the common theme or purpose", help="Goal for discretization")
    parser.add_argument("--initial_labels_embeddings", help="Path to the initial labels embeddings file (required if use_discretization is True)")

    args = parser.parse_args()

    # Run hierarchical clustering
    result = run_hierarchical_clustering(
        args.centroids_file,
        args.output_dir,
        min_clusters=args.min_clusters,
        max_clusters=args.max_clusters,
        min_cluster_size=args.min_cluster_size,
        method=args.method,
        metric=args.metric,
    )
    
    # Label and visualize the tree if requested
    if args.label_tree:
        if args.initial_labels is None:
            print("Warning: No initial labels file provided. Skipping tree labeling.")
        else:
            # Check if discretization is enabled but embeddings are missing
            if args.use_discretization and args.initial_labels_embeddings is None:
                print("Warning: Discretization enabled but no embeddings file provided. Disabling discretization.")
                args.use_discretization = False
                
            label_and_visualize_tree(
                result['tree'],
                args.initial_labels,
                args.output_dir,
                min_descendants=args.min_descendants,
                max_depth=args.max_depth,
                num_samples_to_label=args.num_samples,
                visualize=not args.no_visualize,
                examples_metadata={
                    'examples_dataframe_path': args.examples_df,
                    'num_examples_per_node': args.num_examples_per_node,
                    'cluster_col': args.examples_cluster_col,
                    'sentence_col': args.examples_sentence_col
                },
                use_discretization=args.use_discretization,
                discretization_goal=args.discretization_goal,
                initial_labels_embeddings_path=args.initial_labels_embeddings
            )

if __name__ == "__main__":
    main() 

"""
    python src/step_6__agglomerative_clustering.py \
        experiments/editorial/models/cluster_centroids.npy \
        experiments/editorial/hierarchy_results \
        --min_clusters 2 \
        --max_clusters 15 \
        --min_cluster_size 2 \
        --method ward \
        --metric euclidean \
        --label_tree \
        --initial_labels experiments/editorial/models/cluster_labels.csv \
        --min_descendants 2 \
        --max_depth 8 \
        --num_samples 10  \
        --examples_df experiments/editorial/models/all_extracted_discourse_with_clusters.csv \
        --num_examples_per_node 10 \
        --examples_cluster_col cluster \
        --examples_sentence_col sentences

    # Example with discretization:
    python src/step_6__agglomerative_clustering.py \
        experiments/editorial/models/cluster_centroids.npy \
        experiments/editorial/hierarchy_results_discretized \
        --min_clusters 2 \
        --max_clusters 15 \
        --min_cluster_size 2 \
        --method ward \
        --metric euclidean \
        --label_tree \
        --initial_labels experiments/editorial/models/cluster_labels.csv \
        --min_descendants 2 \
        --max_depth 8 \
        --num_samples 10  \
        --examples_df experiments/editorial/models/all_extracted_discourse_with_clusters.csv \
        --num_examples_per_node 10 \
        --examples_cluster_col cluster \
        --examples_sentence_col sentences \
        --use_discretization \
        --discretization_goal "determine the argumentation strategy" \
        --initial_labels_embeddings experiments/editorial/initial_label_embeddings_cache.npy
"""