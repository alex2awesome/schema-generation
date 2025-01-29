#!/usr/bin/env python
# coding: utf-8

# ----------------------------------------------------------------
""" 
This script processes narrative function data to generate prompts for VLLM and/or theOpenAI API.

    The input CSV file should have the following format:
        1. A column named 'Narrative Function' which contains the text data for which comparisons 
            will be made. ( if it's a different column, use the --col_name argument)
        2. Optionally, a column named 'Is_Error' which indicates whether a row 
           should be excluded from processing (rows with 'No' will be included).
"""

import os
import re
import glob
import json
import argparse
import pandas as pd
import numpy as np
import jsonlines
from tqdm.auto import tqdm
import random
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity
from torch.nn.functional import cosine_similarity as torch_cosine_similarity
from sentence_transformers import SentenceTransformer
import sys
import jsonlines
sys.path.append('..')
import torch
import itertools
from utils_vllm_client import run_vllm_batch, match_batched_vllm_results_to_prompts

# Set environment variable
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def batchify_dataframe(df, batch_size):
    num_batches = len(df) // batch_size + (1 if len(df) % batch_size != 0 else 0)
    return [df.iloc[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]


def compute_embeddings(
        input_df,
        embedding_model_name, 
        col_name='Narrative Function', 
        err_col_name='Is_Error',
        max_returned=None
):
    """
    Compute embeddings for the specified column in the DataFrame using a SentenceTransformer model.
    
    Parameters:
    - source_df: DataFrame containing the source data.
    - embedding_model_name: Name of the SentenceTransformer model to use for computing embeddings.
    - col_name: Name of the column to compute embeddings for (default is 'Narrative Function').
    - err_col_name: Name of the column indicating errors (default is 'Is_Error').
    - max_returned: Maximum number of samples to return (optional).
    
    Returns:
    - embeddings: Computed embeddings for the specified column.
    - idx_of_df: Index of the DataFrame rows used for computing embeddings.
    """
    model = SentenceTransformer(embedding_model_name)
    if err_col_name in input_df.columns:
        input_df = input_df.loc[input_df[err_col_name] == 'No', col_name]
    if max_returned:
        input_df = input_df.sample(n=max_returned)

    narrative_functions = input_df[col_name].dropna()
    texts = narrative_functions.str.split(':').str.get(0).tolist()
    embeddings = model.encode(texts, show_progress_bar=True)
    idx_of_df = narrative_functions.index
    return embeddings, idx_of_df


def sample_from_batch(batch_high_sim_pairs_gpu, sample_size_per_batch):
    # Get the number of rows in the batch
    num_rows = batch_high_sim_pairs_gpu.size(0)
    sample_size_per_batch = min(sample_size_per_batch, num_rows)
    sampled_indices = torch.randperm(num_rows, device=batch_high_sim_pairs_gpu.device)[:sample_size_per_batch]
    sampled_batch = batch_high_sim_pairs_gpu[sampled_indices]
    return sampled_indices, sampled_batch


def compute_high_similarity_pairs(
        embeddings, 
        idx_of_df,
        min_sim_threshold=0.3, 
        max_sim_threshold=0.99999, 
        sample_size=2_000_000, 
        batch_size=50_000
):
    """
    Identify pairs of data points with high similarity based on cosine similarity of embeddings.
    This version processes embeddings in batches to handle large datasets efficiently using GPU.
    
    Parameters:
    - embeddings: Embeddings of the data points.
    - idx_of_df: Index of the DataFrame rows corresponding to the embeddings.
    - min_sim_threshold: Minimum threshold for considering a pair as highly similar (default is 0.3).
    - max_sim_threshold: Maximum threshold for considering a pair as highly similar (default is 0.99999).
    - sample_size: Number of high similarity pairs to sample (default is 2,000,000).
    - batch_size: Number of embeddings to process in each batch (default is 1000).
    
    Returns:
    - high_sim_sample: DataFrame containing sampled high similarity pairs.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embeddings_tensor = torch.tensor(embeddings).to(device)
    idx_of_df_tensor = torch.tensor(idx_of_df).to(device)
    norm_embeddings = embeddings_tensor / embeddings_tensor.norm(dim=1, keepdim=True)
    num_embeddings = len(embeddings)
    high_sim_pairs = []
    num_batches = int(np.ceil(num_embeddings / batch_size))
    sample_size_per_batch = int(np.ceil(sample_size / num_batches))

    # Process embeddings in batches
    for start in tqdm(range(0, num_embeddings, batch_size), total=num_batches, desc='Processing similarity batches'):
        end = min(start + batch_size, num_embeddings)
        batch_idx = idx_of_df_tensor[start:end]
        batch_norm_embeddings = norm_embeddings[start:end]

        # Compute cosine similarity for the current batch using GPU
        similarity_matrix = torch.mm(batch_norm_embeddings, norm_embeddings.t())
        
        # Zero out lower triangle and diagonal for the current batch
        similarities = torch.triu(similarity_matrix, diagonal=1)
        high_sim_indices = torch.nonzero(
            (similarity_matrix > min_sim_threshold) & (similarity_matrix < max_sim_threshold), as_tuple=True
        )
        similarities = similarity_matrix[high_sim_indices]
        
        # No need to move to CPU until sampling
        batch_high_sim_pairs_gpu = torch.stack([
            batch_idx[high_sim_indices[0]],  # row indices from batch
            idx_of_df_tensor[high_sim_indices[1]],  # column indices from original embeddings
            similarities
        ], dim=1)

        # Sample from the batch
        num_high_sim_pairs = batch_high_sim_pairs_gpu.size(0)
        sample_size_this_batch = min(sample_size_per_batch, num_high_sim_pairs)
        if sample_size_this_batch > 0:
            sampled_indices = torch.randperm(num_high_sim_pairs, device=device)[:sample_size_this_batch]
            sampled_batch = batch_high_sim_pairs_gpu[sampled_indices]
            high_sim_pairs.append(sampled_batch.cpu())
    
    all_high_sim_pairs = torch.cat(high_sim_pairs, dim=0)
    all_high_sim_pairs = all_high_sim_pairs.numpy()
    high_sim_df = pd.DataFrame(all_high_sim_pairs, columns=['level_0', 'level_1', 'similarity'])

    print(f"Number of high similarity pairs: {len(high_sim_df)}")
    print(f"Filtering out self-pairs...")
    print(f"Sampling {min(sample_size, len(high_sim_df))} high similarity pairs...")
    high_sim_sample = high_sim_df.sample(n=min(sample_size, len(high_sim_df)))
    return high_sim_sample


def compute_high_similarity_pairs_old(embeddings, idx_of_df, min_sim_threshold=0.3, max_sim_threshold=0.99999, sample_size=2_000_000, batch_size=1000):
    """
    Identify pairs of data points with high similarity based on cosine similarity of embeddings.
    This version processes embeddings in batches to handle large datasets efficiently.
    
    Parameters:
    - embeddings: Embeddings of the data points.
    - idx_of_df: Index of the DataFrame rows corresponding to the embeddings.
    - min_sim_threshold: Minimum threshold for considering a pair as highly similar (default is 0.3).
    - max_sim_threshold: Maximum threshold for considering a pair as highly similar (default is 0.99999).
    - sample_size: Number of high similarity pairs to sample (default is 2,000,000).
    - batch_size: Number of embeddings to process in each batch (default is 1000).
    
    Returns:
    - high_sim_sample: DataFrame containing sampled high similarity pairs.
    """
    num_embeddings = len(embeddings)
    high_sim_pairs = []
    num_batches = int(np.ceil(num_embeddings / batch_size))

    # Process embeddings in batches
    for start in tqdm(range(0, num_embeddings, batch_size), total=num_batches, desc='Processing similarity batches'):
        end = min(start + batch_size, num_embeddings)
        batch_embeddings = embeddings[start:end]
        batch_idx = idx_of_df[start:end]

        # Compute cosine similarity for the current batch
        similarity_matrix = sk_cosine_similarity(batch_embeddings, embeddings)
        
        # Zero out lower triangle and diagonal for the current batch
        similarity_matrix = np.triu(similarity_matrix, k=1)

        # Convert to DataFrame and filter based on similarity threshold
        sim_df = pd.DataFrame(similarity_matrix, index=batch_idx, columns=idx_of_df)
        sim_df = sim_df.stack().reset_index()
        sim_df.columns = ['level_0', 'level_1', 'similarity']
        batch_high_sim_pairs = sim_df.loc[
            (sim_df['similarity'] > min_sim_threshold) &
            (sim_df['similarity'] < max_sim_threshold)
        ]
        batch_high_sim_pairs = batch_high_sim_pairs.loc[batch_high_sim_pairs['level_0'] != batch_high_sim_pairs['level_1']]
        batch_high_sim_pairs = batch_high_sim_pairs.sample(n=int(np.ceil(sample_size / num_batches)) )
        high_sim_pairs.append(batch_high_sim_pairs)

    # Concatenate all high similarity pairs from each batch
    high_sim_pairs = pd.concat(high_sim_pairs, ignore_index=True)
    print(f"Number of high similarity pairs: {len(high_sim_pairs)}")
    print(f"Filtering out self-pairs...")
    print(f"Sampling {min(sample_size, len(high_sim_pairs))} high similarity pairs...")
    high_sim_sample = high_sim_pairs.sample(n=min(sample_size, len(high_sim_pairs)))

    return high_sim_sample


def create_high_similarity_samples(text_df, high_sim_sample, text_col_name='Narrative Function'):
    """
    Create a DataFrame with pairs of texts that have high similarity.
    
    Parameters:
    - text_df: DataFrame containing the text data.
    - high_sim_sample: DataFrame containing high similarity pairs.
    - text_col_name: Name of the column containing the text data (default is 'Narrative Function').
    
    Returns:
    - high_sim_pairwise_samples_to_evaluate: DataFrame with pairs of high similarity texts.
    """
    narrative_functions = text_df[text_col_name]
    high_sim_pairwise_samples_to_evaluate = (
        pd.concat([
            narrative_functions.loc[high_sim_sample['level_0']].reset_index(drop=True).rename('description_1'),
            narrative_functions.loc[high_sim_sample['level_1']].reset_index(drop=True).rename('description_2'),
        ], axis=1)
        .dropna()
        .drop_duplicates()
    )
    return high_sim_pairwise_samples_to_evaluate


def generate_prompts(high_sim_samples, k=3):
    """
    Generate prompts for the OpenAI API based on high similarity samples.
    
    Parameters:
    - high_sim_samples: DataFrame containing high similarity text pairs.
    - k: Number of pairs to include in each prompt (default is 5).
    
    Returns:
    - all_prompts: List of generated prompts for the OpenAI API.
    """
    all_prompts = []
    all_batched_inputs = []
    total_batches = int(len(high_sim_samples) / k)
    for t in tqdm(batchify_dataframe(high_sim_samples, k), total=total_batches, desc='Generating prompts'):
        samples = []
        for i, (_, row) in enumerate(t.iterrows(), 1):
            line = f'{i}. Description 1: {row["description_1"]}, Description 2: {row["description_2"]}'
            samples.append(line)
        
        k_i = min(k, len(t))
        prompt = (f"""I will show you {k_i} pairs of of outline bullet-points, all from different video script outlines.

        Are the two bullet-points in each pair describing similar narrative roles in their respective outlines?
        Think broadly about the role each bullet-point is describing in the pair. Don't pay attention to the specific events of each bullet point.
        Answer with "Yes" or "No". Answer each sequentially and number them with 1., 2., 3., etc.

        [Short Example]
        1. Description 1: ```Volcano Example. Illustrates the concept with a real-world example.. Engages the audience with a surprising fact about volcanic eruptions.```, Description 2: ```Flat Earth Hypothesis. Explores the hypothetical scenario of a flat Earth.. Provides a detailed explanation of how gravity would work on a flat Earth.```
        2. Description 1: ```- **3.5**: FFT Algorithm. Provides a detailed explanation of the FFT algorithm.. Educates the audience on the technical details of FFT.```, Description 2: ```- **6.1. Malfunction 54**, Malfunction 54. Explains the specific error and its implications.. Details the critical error that led to multiple incidents.```
        3. Description 1: ```2. The Incident, The Incident. Describes the critical event that changes the course of the mission.. Introduces the main conflict and sets up the stakes.```, Description 2: ```3. Initial Attempt and Failure, Initial Attempt and Failure. Documents the first attempt and its failure.. Demonstrates the trial-and-error process in scientific experiments.```
        4. Description 1: ```Call to Action. Encourages viewers to share their experiences and comment.. Promotes audience interaction and community building.```, Description 2: ```Conclusion. Wraps up the video, encourages engagement, and promotes the channel.. Provides closure and calls to action for viewers.```
        Answers: 1. No, 2. No, 3. Yes, 4. Yes
                  
        Now it's your turn:
        """ + 
        '\n'.join(samples) +
        """
        Answers:
        """)
        all_prompts.append(prompt)
        all_batched_inputs.append(t)
    return all_prompts, all_batched_inputs


def create_triplets(full_data_exp_df, max_negatives_per_positive=10, max_positives=10):
    """
    Creates triplets of the form {'anchor': description_1, 'positive': description_2_positive, 'negative': description_2_negative}
    where each description_1 is paired with a description_2 with "Yes" (positive) and "No" (negative).
    
    Parameters:
        full_data_exp_df (pd.DataFrame): DataFrame containing 'description_1', 'description_2', and 'output' columns.
    
    Returns:
        triplets (list): List of dictionaries containing 'anchor', 'positive', and 'negative' keys.
    """
    # Initialize an empty list to collect triplets
    triplets = []

    # Ensure the 'output' column is in a consistent format
    full_data_exp_df['label'] = full_data_exp_df['label'].str.strip().str.lower()

    # Group the DataFrame by 'description_1'
    grouped = full_data_exp_df.groupby('description_1')

    # For each group (each 'description_1')
    for description_1_value, group in grouped:
        # Get the positive and negative examples
        positive_examples = group.loc[group['label'] == 'yes', 'description_2'].tolist()
        negative_examples = group.loc[group['label'] == 'no', 'description_2'].tolist()

        # If we have at least one positive and one negative example
        if positive_examples and negative_examples:
            # For each positive example
            # Optionally limit the number of negatives per positive
            if max_positives:
                positives_to_use = random.sample(positive_examples, min(max_positives, len(positive_examples)))
            else:
                positives_to_use = positive_examples

            if max_negatives_per_positive:
                negatives_to_use = random.sample(negative_examples, min(max_negatives_per_positive, len(negative_examples)))
            else:
                negatives_to_use = negative_examples
                
            for pos in positives_to_use:
                for neg in negatives_to_use:
                    # Create triplet
                    triplet = {
                        'anchor': description_1_value,
                        'positive': pos,
                        'negative': neg
                    }
                    # Append to list
                    triplets.append(triplet)
                    
    return triplets


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Process narrative function similarity.')
    parser.add_argument('--input_file', type=str, default='../data/v2_narr_keywords', help='Directory containing the input data files.')
    parser.add_argument('--output_dir', type=str, default='../data/openai-batches/narr-role-similarity', help='Directory to save output files.')
    parser.add_argument('--embedding_model_name', type=str, default='all-MiniLM-L6-v2', help='SentenceTransformer model name for embeddings.')
    parser.add_argument('--model_name', type=str, default="meta-llama/Meta-Llama-3.1-70B-Instruct", help='Model name for prompting an LLM.')
    parser.add_argument('--col_name', type=str, default='Narrative Function', help='Column name to compute embeddings for.')
    parser.add_argument('--min_sim_threshold', type=float, default=0.3, help='Similarity threshold for selecting pairs.')
    parser.add_argument('--max_sim_threshold', type=float, default=0.9, help='Similarity threshold for selecting pairs.')
    parser.add_argument('--sample_size', type=int, default=2000000, help='Number of sample pairs to process.')
    parser.add_argument('--k', type=int, default=5, help='Number of pairs per prompt.')
    parser.add_argument('--batch_size', type=int, default=40000, help='Number of prompts per batch file.')
    parser.add_argument('--completion_window', type=str, default='24h', help='Completion window for OpenAI batch processing.')
    parser.add_argument('--output_file', type=str, default='../data/openai-batches/narr-role-similarity/processed_narrative_roles.csv', help='Output file path.')
    parser.add_argument('--debug', action='store_true', help='Debug mode.')
    args = parser.parse_args()

    # Process source data
    input_df = pd.read_csv(args.input_file)

    #
    # Step 1: create data for training 
    # ----------------------------------------------------------------
    # Compute embeddings
    embeddings, idx_of_df = compute_embeddings(input_df, args.embedding_model_name, col_name=args.col_name)
    if args.debug:
        embeddings = embeddings[:10000]
        idx_of_df = idx_of_df[:10000]
    
    high_sim_sample = compute_high_similarity_pairs(
        embeddings, 
        idx_of_df, 
        args.min_sim_threshold, 
        args.max_sim_threshold, 
        args.sample_size,
        args.batch_size
    )
    high_sim_samples = create_high_similarity_samples(input_df, high_sim_sample, text_col_name=args.col_name)

    #
    # Step 2: Generate prompts for prompting an LLM to label the data
    # ----------------------------------------------------------------
    all_prompts, all_batched_inputs = generate_prompts(high_sim_samples, k=args.k)
    if args.debug:
        with open('debug_prompts.txt', 'w') as f:
            f.write('\n'.join(all_prompts))

    results = run_vllm_batch(all_prompts, args.model_name)
    full_data_exp_df = match_batched_vllm_results_to_prompts(results, all_batched_inputs)
    output_path, output_filename = os.path.split(args.output_file)
    output_fname, output_ext = os.path.splitext(output_filename)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_filename = f'paired_labeled_data_{output_fname}.csv'
    full_data_exp_df.to_csv(os.path.join(output_path, output_filename), index=False)

    # Step 3: Create triplets and save to file
    # ----------------------------------------------------------------
    triplets = create_triplets(full_data_exp_df)
    output_filename = f'triplets_{output_fname}.jsonl'
    with jsonlines.open(os.path.join(output_path, output_filename), 'w') as f:
        f.write_all(triplets)
        
    print(f"Processing complete. Data saved to '{args.output_file}'.")

if __name__ == "__main__":
    main()

