---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:75013
- loss:MultipleNegativesRankingLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: '"Disappointment": Feeling let down by the situation.'
  sentences:
  - '"Disbelief": Struggling to accept the truth of the situation.'
  - '"Shock": Surprised by the statement or situation.'
  - '"Frustration": The commenter expresses frustration about the craziness of the
    situation.'
- source_sentence: '"Amusement": The commenter finds humor in the reactions of others.'
  sentences:
  - '"Confidence": The commenter expresses confidence in their own beliefs.'
  - '"Excitement": The commenter expresses excitement about sharing music with their
    spouse.'
  - '"Frustration": Expressing intense frustration or anger.'
- source_sentence: '"Amusement": The commenter finds the situation funny.'
  sentences:
  - '"Support": The commenter expresses support and encouragement for a service.'
  - '"Indifference": The commenter shows a lack of concern or interest in the negativity
    expressed by another.'
  - '"Playfulness": Engaging in a light-hearted or humorous manner.'
- source_sentence: '"Disapproval": Expressing negative judgment about someone''s character.'
  sentences:
  - '"Moral Outrage": Expressing strong disapproval of unethical behavior.'
  - '"Neutral": No strong emotions expressed, just a statement.'
  - '"Nostalgia": Reflecting fondly on a past experience.'
- source_sentence: '"Sarcasm": The commenter expresses a sarcastic tone about their
    success.'
  sentences:
  - '"Humor": The commenter uses humor in their comment.'
  - '"Gratitude": The commenter expresses appreciation for the explanation, indicating
    a positive emotional response.'
  - '"Admiration": The commenter expresses respect for the skills or abilities of
    others.'
pipeline_tag: sentence-similarity
library_name: sentence-transformers
metrics:
- cosine_accuracy
model-index:
- name: SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2
  results:
  - task:
      type: triplet
      name: Triplet
    dataset:
      name: all nli dev
      type: all-nli-dev
    metrics:
    - type: cosine_accuracy
      value: 0.7347330451011658
      name: Cosine Accuracy
  - task:
      type: triplet
      name: Triplet
    dataset:
      name: all nli dev trainer
      type: all-nli-dev-trainer
    metrics:
    - type: cosine_accuracy
      value: 0.9500899910926819
      name: Cosine Accuracy
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) on the json dataset. It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision c9745ed1d9f207416be6d2e6f8de32d1f16199bf -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
- **Training Dataset:**
    - json
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    '"Sarcasm": The commenter expresses a sarcastic tone about their success.',
    '"Humor": The commenter uses humor in their comment.',
    '"Admiration": The commenter expresses respect for the skills or abilities of others.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Triplet

* Datasets: `all-nli-dev` and `all-nli-dev-trainer`
* Evaluated with [<code>TripletEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.TripletEvaluator)

| Metric              | all-nli-dev | all-nli-dev-trainer |
|:--------------------|:------------|:--------------------|
| **cosine_accuracy** | **0.7347**  | **0.9501**          |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### json

* Dataset: json
* Size: 75,013 training samples
* Columns: <code>anchor</code>, <code>positive</code>, and <code>negative</code>
* Approximate statistics based on the first 1000 samples:
  |         | anchor                                                                             | positive                                                                           | negative                                                                           |
  |:--------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
  | type    | string                                                                             | string                                                                             | string                                                                             |
  | details | <ul><li>min: 11 tokens</li><li>mean: 17.67 tokens</li><li>max: 31 tokens</li></ul> | <ul><li>min: 11 tokens</li><li>mean: 18.27 tokens</li><li>max: 31 tokens</li></ul> | <ul><li>min: 11 tokens</li><li>mean: 18.69 tokens</li><li>max: 30 tokens</li></ul> |
* Samples:
  | anchor                                                                                                         | positive                                                                                                | negative                                                                                                  |
  |:---------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------|
  | <code>"Amusement": The commenter finds humor in the situation, indicating a light-hearted attitude.</code>     | <code>"Playfulness": Engaging in a light-hearted suggestion.</code>                                     | <code>"Sadness": The commenter feels a sense of sorrow about the truth of the statement.</code>           |
  | <code>"Frustration": The commenter feels frustrated about the lack of recognition or hype for the show.</code> | <code>"Frustration": The commenter expresses annoyance regarding distractions in a team setting.</code> | <code>"Curiosity": The commenter is expressing curiosity about the presence of a political figure.</code> |
  | <code>"Anger": The commenter expresses strong negative feelings towards a group.</code>                        | <code>"Sarcasm": Using sarcasm to criticize someone's lack of sensitivity.</code>                       | <code>"Playful": Light-hearted tone in the comment.</code>                                                |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim"
  }
  ```

### Evaluation Dataset

#### json

* Dataset: json
* Size: 8,335 evaluation samples
* Columns: <code>anchor</code>, <code>positive</code>, and <code>negative</code>
* Approximate statistics based on the first 1000 samples:
  |         | anchor                                                                            | positive                                                                           | negative                                                                          |
  |:--------|:----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|
  | type    | string                                                                            | string                                                                             | string                                                                            |
  | details | <ul><li>min: 9 tokens</li><li>mean: 17.84 tokens</li><li>max: 31 tokens</li></ul> | <ul><li>min: 10 tokens</li><li>mean: 18.13 tokens</li><li>max: 31 tokens</li></ul> | <ul><li>min: 9 tokens</li><li>mean: 18.62 tokens</li><li>max: 36 tokens</li></ul> |
* Samples:
  | anchor                                                                                              | positive                                                                                            | negative                                                                                                       |
  |:----------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------|
  | <code>"Disapproval": The commenter expresses disapproval of the violent situation described.</code> | <code>"Disapproval": The commenter is expressing negative judgment towards someone.</code>          | <code>"Sympathy": Feeling sorry for someone in a difficult situation.</code>                                   |
  | <code>"Disappointment": The commenter feels let down by the person's actions.</code>                | <code>"Frustration": Feeling annoyed by the conflicting opinions on sensitive topics.</code>        | <code>"Hopefulness": The commenter expresses a hopeful outlook on their future and personal well-being.</code> |
  | <code>"Shock": The commenter expresses disbelief or horror at the situation described.</code>       | <code>"Frustration": The commenter expresses annoyance and impatience regarding a situation.</code> | <code>"Frustration": The strong negative sentiment towards the manager suggests frustration.</code>            |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: epoch
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 32
- `learning_rate`: 2e-05
- `num_train_epochs`: 2
- `warmup_ratio`: 0.1
- `ddp_backend`: nccl
- `ddp_find_unused_parameters`: True
- `batch_sampler`: no_duplicates

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: epoch
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 32
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 2e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1.0
- `num_train_epochs`: 2
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.1
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 1
- `ddp_backend`: nccl
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: True
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `tp_size`: 0
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: True
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: no_duplicates
- `multi_dataset_batch_sampler`: proportional

</details>

### Training Logs
| Epoch  | Step | Training Loss | Validation Loss | all-nli-dev_cosine_accuracy | all-nli-dev-trainer_cosine_accuracy |
|:------:|:----:|:-------------:|:---------------:|:---------------------------:|:-----------------------------------:|
| -1     | -1   | -             | -               | 0.7347                      | -                                   |
| 0.0853 | 100  | 3.4219        | -               | -                           | -                                   |
| 0.1706 | 200  | 2.6979        | -               | -                           | -                                   |
| 0.2560 | 300  | 2.5798        | -               | -                           | -                                   |
| 0.3413 | 400  | 2.5368        | -               | -                           | -                                   |
| 0.4266 | 500  | 2.5074        | -               | -                           | -                                   |
| 0.5119 | 600  | 2.507         | -               | -                           | -                                   |
| 0.5973 | 700  | 2.4772        | -               | -                           | -                                   |
| 0.6826 | 800  | 2.464         | -               | -                           | -                                   |
| 0.7679 | 900  | 2.4625        | -               | -                           | -                                   |
| 0.8532 | 1000 | 2.4497        | -               | -                           | -                                   |
| 0.9386 | 1100 | 2.4401        | -               | -                           | -                                   |
| 1.0    | 1172 | -             | 3.1210          | -                           | 0.9467                              |
| 1.0239 | 1200 | 2.4306        | -               | -                           | -                                   |
| 1.1092 | 1300 | 2.4417        | -               | -                           | -                                   |
| 1.1945 | 1400 | 2.4197        | -               | -                           | -                                   |
| 1.2799 | 1500 | 2.4117        | -               | -                           | -                                   |
| 1.3652 | 1600 | 2.4036        | -               | -                           | -                                   |
| 1.4505 | 1700 | 2.3716        | -               | -                           | -                                   |
| 1.5358 | 1800 | 2.4095        | -               | -                           | -                                   |
| 1.6212 | 1900 | 2.3803        | -               | -                           | -                                   |
| 1.7065 | 2000 | 2.3889        | -               | -                           | -                                   |
| 1.7918 | 2100 | 2.3736        | -               | -                           | -                                   |
| 1.8771 | 2200 | 2.3864        | -               | -                           | -                                   |
| 1.9625 | 2300 | 2.367         | -               | -                           | -                                   |
| 2.0    | 2344 | -             | 3.0694          | -                           | 0.9501                              |


### Framework Versions
- Python: 3.11.8
- Sentence Transformers: 4.1.0
- Transformers: 4.51.3
- PyTorch: 2.6.0+cu124
- Accelerate: 1.0.1
- Datasets: 3.6.0
- Tokenizers: 0.21.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->