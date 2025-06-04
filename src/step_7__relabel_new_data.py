"""
This script takes:
- a dataframe with a column of sentences
- a dataframe with the following information:
    - a column of node levels 
    - a column of labels
    - a column of descriptions per label
- a node level specification 

It then prompts an LLM to relabel the sentences in the first dataframe using the label definitions/descriptions from the second dataframe.  

The node level specification is a string that specifies the node level to use for relabeling.
"""

from typing import Annotated, Literal
import pandas as pd
from pydantic import create_model, Field
from typing import Annotated, Literal
from annotated_types import Len
import os
from tqdm.auto import tqdm
from more_itertools import flatten

# Only set vLLM environment variables if we're using vLLM
if not any('--use_openai' in arg for arg in os.sys.argv):
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'
    os.environ["MKL_THREADING_LAYER"] = "GNU"
    os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
    from utils_vllm_client import load_model as load_vllm_model
    from vllm import LLM, SamplingParams, run_vllm_batch
    USE_OPENAI = False
else:
    from utils_openai_client import prompt_openai_model
    USE_OPENAI = True


def make_labeling_structure(possible_labels: list[str]):
    return create_model(
        'LabelingResponse',
        label=(Literal[tuple(possible_labels)], ...) # type: ignore
    )


def make_multi_sentence_labeling_structure(possible_labels: list[str], num_sentences: int):
    return create_model(
        'MultiLabelingResponse',
        labels=(Annotated[
            list[Literal[tuple(possible_labels)]], # type: ignore
            Len(min_length=num_sentences, max_length=num_sentences)
        ], ...)
    )

def run_openai_batch(
        batch: pd.DataFrame,
        labels: list[str],
        descriptions: list[str],
        model: str,
        goal: str,
        text_col: str,
):
    batch_size = len(batch)
    if batch_size == 1:
        response_object = make_labeling_structure(labels)
    else:
        response_object = make_multi_sentence_labeling_structure(labels, batch_size)

    labels_and_descriptions = "\n".join([f"{label}: {description}" for label, description in zip(labels, descriptions)])
    sentences = "\n\n".join(batch[text_col].tolist())
    # prompt the model
    prompt = """
You are a helpful assistant that labels sentences. I will show you a list of {num_labels} labels and their descriptions.

<labels_and_descriptions>
{labels_and_descriptions}
</labels_and_descriptions>

Now here are {num_sentences} sentences.

<sentences>
{sentences}
</sentences>

For each sentence, choose the label that best captures what the sentence is. Return one label per sentence.

Your response:
"""

    prompt = prompt.format(
        goal=goal,
        labels_and_descriptions=labels_and_descriptions,
        sentences=sentences,
        num_labels=len(labels),
        num_sentences=batch_size,
    )
    response = prompt_openai_model(
        prompt=prompt, 
        model_name=model, 
        response_format=response_object,
    )
    return response.labels if batch_size > 1 else [response.label]


def label_data_in_batches(
        data_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        node_level_specification: str,
        model: str,
        batch_size: int,
        goal: str,
        text_col: str,
        label_col: str,
        description_col: str,
        level_col: str,
):
    # get the labels for the node level
    labels = labels_df.loc[lambda df: df[level_col] == node_level_specification, label_col].tolist()
    descriptions = labels_df.loc[lambda df: df[level_col] == node_level_specification, description_col].tolist()

    # get the data in batches
    batches = [data_df.iloc[i:i+batch_size] for i in range(0, len(data_df), batch_size)]

    # relabel the data
    relabeled_data = []
    for batch in tqdm(batches, total=len(batches), desc='Labeling data...'):
        if USE_OPENAI:
            relabeled_data.append(run_openai_batch(batch, labels, descriptions, model, goal, text_col))
        else:
            # todo: finish implementing this
            relabeled_data.append(run_vllm_batch(batch, labels, descriptions, model, goal, text_col))

    data = list(flatten(relabeled_data))
    data_df['labels'] = data
    return data_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_file", type=str, required=True)
    parser.add_argument("--labels_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--text_col", type=str, default='sentence')
    parser.add_argument("--label_col", type=str, default='label')
    parser.add_argument("--description_col", type=str, default='description')
    parser.add_argument("--level_col", type=str, default='level')
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--node_level_specification", type=int, default=None)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--use_openai", action="store_true")
    parser.add_argument("--goal", type=str, required=True)
    args = parser.parse_args()

    # load the data
    labels_df = pd.read_csv(args.labels_file)
    data_df = pd.read_csv(args.input_data_file)
    if args.node_level_specification is not None:
        node_level_specification = [args.node_level_specification]
    else:
        node_level_specification = labels_df[args.level_col].unique().tolist()

    for node_level in tqdm(node_level_specification, total=len(node_level_specification), desc='Node levels...'):
        relabeled_data_df = label_data_in_batches(
            data_df=data_df,
            labels_df=labels_df,
            text_col=args.text_col,
            label_col=args.label_col,
            description_col=args.description_col,
            level_col=args.level_col,
            node_level_specification=node_level,
            model=args.model,
            batch_size=args.batch_size,
            goal=args.goal,
        )

        # save the data
        output_file, output_file_ext = os.path.splitext(args.output_file)
        output_file = f'{output_file}__level-{node_level}{output_file_ext}'
        print(f'Saving to {output_file}...')
        relabeled_data_df.to_csv(output_file, index=False)


"""
Generic command:
python src/step_7__relabel_new_data.py \
  --input_data_file path/to/input_data.csv \
  --labels_file path/to/labels.csv \
  --output_file path/to/output.csv \
  --model your-model-name \
  --node_level_specification your_node_level \
  --batch_size 8 \
  --goal "Your labeling goal here" \
  --use_openai

  
# reasoning
python src/step_7__relabel_new_data.py \
  --input_data_file experiments/reasoning/qwq-32b/make_hierarchy/chunk_sample_exploded.csv.gz \
  --labels_file experiments/reasoning/qwq-32b/models/chess_agglomerative_clustering_outputs__discretized__labels_descriptions__child-nodes__output-labels-desc/labels_and_descriptions.csv \
  --output_file experiments/reasoning/qwq-32b/models/chess_agglomerative_clustering_outputs__discretized__labels_descriptions__child-nodes__output-labels-desc/labeled_chunks.csv \
  --model gpt-4o \
  --node_level_specification "level_2" \
  --batch_size 8 \
  --goal "identify the nature of the mathematical reasoning being executed in the chunk" \
  --use_openai
"""    