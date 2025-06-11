---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:100000
- loss:MultipleNegativesRankingLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: '"Analysis": Providing a critical perspective on relationships.'
  sentences:
  - '"Personal Experience": Sharing personal preference to counter the stereotype.'
  - '"Inquisitive": Seeking more information in a neutral manner.'
  - '"Judgmental": Criticizing the woman without context.'
- source_sentence: '"Humorous": Making a light-hearted comment related to the original.'
  sentences:
  - '"Dismissive": Short response that ignores the original message.'
  - '"Dismissive": Dismissing the argument with sarcasm.'
  - '"Accusatory": Labeling the other as extreme or hateful.'
- source_sentence: '"Sarcastic": Using irony to undermine the original violent suggestion.'
  sentences:
  - '"Mocking": Making fun of the original comment.'
  - '"Agreeing": Concur with the sentiment expressed.'
  - '"Sarcastic": Using sarcasm to critique the opposing argument.'
- source_sentence: '"Questioning": Challenging the original comment''s assertion.'
  sentences:
  - '"Confrontational": Challenges the original poster''s viewpoint.'
  - '"Clarification": Attempts to specify details in a confrontational manner.'
  - '"Defensive": Defending the user who provided examples.'
- source_sentence: '"Agreement": The response agrees with the sentiment of the original
    message.'
  sentences:
  - '"Dismissive": Nonchalant response to the original message.'
  - '"Skeptical": Questioning the logic of the original statement.'
  - '"Challenge": Questioning the original claim.'
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
      value: 0.7193551659584045
      name: Cosine Accuracy
  - task:
      type: triplet
      name: Triplet
    dataset:
      name: all nli dev trainer
      type: all-nli-dev-trainer
    metrics:
    - type: cosine_accuracy
      value: 0.927329421043396
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
    '"Agreement": The response agrees with the sentiment of the original message.',
    '"Dismissive": Nonchalant response to the original message.',
    '"Challenge": Questioning the original claim.',
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
| **cosine_accuracy** | **0.7194**  | **0.9273**          |

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
* Size: 100,000 training samples
* Columns: <code>anchor</code>, <code>positive</code>, and <code>negative</code>
* Approximate statistics based on the first 1000 samples:
  |         | anchor                                                                             | positive                                                                           | negative                                                                          |
  |:--------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|
  | type    | string                                                                             | string                                                                             | string                                                                            |
  | details | <ul><li>min: 10 tokens</li><li>mean: 15.56 tokens</li><li>max: 25 tokens</li></ul> | <ul><li>min: 10 tokens</li><li>mean: 15.41 tokens</li><li>max: 24 tokens</li></ul> | <ul><li>min: 9 tokens</li><li>mean: 15.49 tokens</li><li>max: 26 tokens</li></ul> |
* Samples:
  | anchor                                                                         | positive                                                                             | negative                                                                          |
  |:-------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|
  | <code>"Caution": Advice to refrain from premature judgments.</code>            | <code>"Dismissive": Minimizing the outrage expressed in the original message.</code> | <code>"Dismissive": Minimizing the concern raised in the original message.</code> |
  | <code>"Dismissive": Minimizing the seriousness of the original message.</code> | <code>"Indifferent": Shows a lack of concern or engagement with the topic.</code>    | <code>"Aggressive": Using aggressive language towards the subject.</code>         |
  | <code>"Questioning": Challenging the original comment's perspective.</code>    | <code>"Analytical": Providing a thoughtful critique of the original comment.</code>  | <code>"Empathy Appeal": Encourages compassion and understanding.</code>           |
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
* Size: 39,452 evaluation samples
* Columns: <code>anchor</code>, <code>positive</code>, and <code>negative</code>
* Approximate statistics based on the first 1000 samples:
  |         | anchor                                                                             | positive                                                                           | negative                                                                          |
  |:--------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|
  | type    | string                                                                             | string                                                                             | string                                                                            |
  | details | <ul><li>min: 10 tokens</li><li>mean: 15.56 tokens</li><li>max: 23 tokens</li></ul> | <ul><li>min: 10 tokens</li><li>mean: 15.45 tokens</li><li>max: 24 tokens</li></ul> | <ul><li>min: 9 tokens</li><li>mean: 15.57 tokens</li><li>max: 26 tokens</li></ul> |
* Samples:
  | anchor                                                                  | positive                                                                  | negative                                                                                |
  |:------------------------------------------------------------------------|:--------------------------------------------------------------------------|:----------------------------------------------------------------------------------------|
  | <code>"Dismissive": Ignoring the original question with sarcasm.</code> | <code>"Aggressive": Attacking the original commenter with insults.</code> | <code>"Defensive": Defending the original statement against perceived criticism.</code> |
  | <code>"Derogatory": Using a pejorative term to insult someone.</code>   | <code>"Dismissive": Belittling the original comment.</code>               | <code>"Deflection": Redirecting the conversation to a personal anecdote.</code>         |
  | <code>"Defensive": Justifying behavior based on attractiveness.</code>  | <code>"Provocative": Challenges the accuser's assumptions.</code>         | <code>"Critique": Pointing out hypocrisy in the original comment.</code>                |
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
- `local_rank`: 3
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
| -1     | -1   | -             | -               | 0.7194                      | -                                   |
| 0.0640 | 100  | 3.9336        | -               | -                           | -                                   |
| 0.1280 | 200  | 3.0787        | -               | -                           | -                                   |
| 0.1921 | 300  | 3.0038        | -               | -                           | -                                   |
| 0.2561 | 400  | 2.9493        | -               | -                           | -                                   |
| 0.3201 | 500  | 2.9414        | -               | -                           | -                                   |
| 0.3841 | 600  | 2.9012        | -               | -                           | -                                   |
| 0.4481 | 700  | 2.8947        | -               | -                           | -                                   |
| 0.5122 | 800  | 2.8878        | -               | -                           | -                                   |
| 0.5762 | 900  | 2.8689        | -               | -                           | -                                   |
| 0.6402 | 1000 | 2.8706        | -               | -                           | -                                   |
| 0.7042 | 1100 | 2.8802        | -               | -                           | -                                   |
| 0.7682 | 1200 | 2.8606        | -               | -                           | -                                   |
| 0.8323 | 1300 | 2.8603        | -               | -                           | -                                   |
| 0.8963 | 1400 | 2.8475        | -               | -                           | -                                   |
| 0.9603 | 1500 | 2.8375        | -               | -                           | -                                   |
| 1.0    | 1562 | -             | 3.5097          | -                           | 0.9200                              |
| 1.0243 | 1600 | 2.8401        | -               | -                           | -                                   |
| 1.0883 | 1700 | 2.8451        | -               | -                           | -                                   |
| 1.1524 | 1800 | 2.8266        | -               | -                           | -                                   |
| 1.2164 | 1900 | 2.8224        | -               | -                           | -                                   |
| 1.2804 | 2000 | 2.8397        | -               | -                           | -                                   |
| 1.3444 | 2100 | 2.8199        | -               | -                           | -                                   |
| 1.4085 | 2200 | 2.8142        | -               | -                           | -                                   |
| 1.4725 | 2300 | 2.8131        | -               | -                           | -                                   |
| 1.5365 | 2400 | 2.8202        | -               | -                           | -                                   |
| 1.6005 | 2500 | 2.8127        | -               | -                           | -                                   |
| 1.6645 | 2600 | 2.8066        | -               | -                           | -                                   |
| 1.7286 | 2700 | 2.7947        | -               | -                           | -                                   |
| 1.7926 | 2800 | 2.8037        | -               | -                           | -                                   |
| 1.8566 | 2900 | 2.7891        | -               | -                           | -                                   |
| 1.9206 | 3000 | 2.8105        | -               | -                           | -                                   |
| 1.9846 | 3100 | 2.8065        | -               | -                           | -                                   |
| 2.0    | 3124 | -             | 3.4666          | -                           | 0.9273                              |


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