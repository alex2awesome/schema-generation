import pandas as pd 
from tqdm.auto import tqdm
import json, os
import logging
from prompts import KEYWORD_DEFINITION_NODE_PROMPT
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(lineno)d - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

# Set up environment variables
here = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(f'{here}/../config.json'):
    config_data = json.load(open(f'{here}/../config.json'))
    os.environ['HF_TOKEN'] = config_data["HF_TOKEN"]

# Only set vLLM environment variables if we're using vLLM
if not any('--use_openai' in arg for arg in os.sys.argv):
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'
    os.environ["MKL_THREADING_LAYER"] = "GNU"
    os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
    from utils_vllm_client import load_model as load_vllm_model
    from vllm import LLM, SamplingParams
else:
    from utils_openai_client import prompt_openai_model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-70B-Instruct")
    parser.add_argument("--index_col", type=str, default='index')
    parser.add_argument("--cluster_col", type=str, default='cluster')
    parser.add_argument("--label_superset_col", type=str, default='output')
    parser.add_argument("--n_samples_per_cluster", type=int, default=10)
    parser.add_argument("--use_openai", action='store_true', help='Use OpenAI API instead of vLLM')
    args = parser.parse_args()

    orig_kmeans = pd.read_csv(args.input_file)
    cluster_inspection = (
        orig_kmeans
            .loc[lambda df: df[args.label_superset_col].notnull()] ## why are some of these labels null?
            .groupby(args.cluster_col)
            .apply(lambda df: list(df[args.label_superset_col].sample(min(args.n_samples_per_cluster, len(df)))))
    )
    logging.info(f'loaded {len(cluster_inspection)} clusters')
    
    prompts = []
    for i, g in tqdm(cluster_inspection.items(), total=len(cluster_inspection)):
        prompt = KEYWORD_DEFINITION_NODE_PROMPT.format(labels='\n'.join(g))
        prompts.append(prompt)
    logging.info(f'generated {len(prompts)} prompts')
    
    initial_node_labels = {}
    
    if args.use_openai:
        logging.info(f'Using OpenAI API with model {args.model}')
        for i, prompt in tqdm(enumerate(prompts), total=len(prompts)):
            response = prompt_openai_model(prompt, model_name=args.model)
            initial_node_labels[i] = response
    else:
        logging.info(f'Using vLLM with model {args.model}')
        sampling_params = SamplingParams(temperature=0.1, max_tokens=256)
        tokenizer, model = load_vllm_model(args.model)
        outputs = model.generate(prompts, sampling_params)
        outputs = sorted(outputs, key=lambda x: int(x.request_id))
        for i, output in enumerate(outputs):
            initial_node_labels[i] = output.outputs[0].text

    initial_node_label_df = (
        pd.Series(initial_node_labels)
            .to_frame('label')
            .reset_index()
            .rename(columns={args.index_col: 'node_id'})
    )
    initial_node_label_df.to_csv(args.output_file, index=False)


"""
python src/label_low_level_kmeans_clusters.py \
    --input_file experiments/editorial/models/all_extracted_discourse_with_clusters.csv \
    --output_file experiments/editorial/models/cluster_labels.csv \
    --model "meta-llama/Meta-Llama-3.1-70B-Instruct" \
    --cluster_col cluster \
    --label_superset_col label \
    --n_samples_per_cluster 10

# To use OpenAI instead:
python src/label_low_level_kmeans_clusters.py \
    --input_file experiments/editorial/models/all_extracted_discourse_with_clusters.csv \
    --output_file experiments/editorial/initial_clustered_node_labels.csv \
    --model "gpt-4o-mini" \
    --cluster_col cluster \
    --label_superset_col label \
    --n_samples_per_cluster 10 \
    --use_openai
"""