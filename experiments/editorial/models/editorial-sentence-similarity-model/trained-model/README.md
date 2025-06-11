---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:150000
- loss:MultipleNegativesRankingLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: '"Implied Causation": This sentence implies a causal relationship
    between the political accusations against Ismayilova and the subsequent actions
    taken against the radio station, reinforcing the narrative of systematic repression.'
  sentences:
  - '"Lack of Oversight": This sentence points out the absence of legal limits on
    classification, highlighting a significant gap in oversight that contributes to
    the problem of accountability.'
  - '"Evidence of Financial Ties": This sentence highlights Hillary Clinton''s financial
    connections to Wall Street, using specific evidence from a reputable source to
    suggest her potential conflict of interest and discomfort with progressive policies.'
  - '"Causal Argument": The author claims that the actions taken by Senator Church
    had long-lasting detrimental effects on intelligence operations, reinforcing the
    argument against releasing the report.'
- source_sentence: '"Responsibility": This sentence emphasizes the dual responsibility
    of politicians, highlighting their obligation to the society they come from while
    acknowledging their greater role in leading the nation.'
  sentences:
  - '"Vision for Renewal": The author concludes with a hopeful vision, advocating
    for a revival of the American spirit through transformative changes in attitudes,
    positioning this as essential for overcoming current challenges.'
  - '"Call to Action": This sentence serves as a call to action, urging Iraqi citizens
    to reflect on their national identity and the viability of their statehood, thereby
    prompting a critical evaluation of their political situation.'
  - '"Consequential Evidence": This sentence highlights the ongoing repercussions
    of the media''s past decisions, suggesting that the threat to safety persists,
    thereby reinforcing the argument that media cowardice has real-world implications.'
- source_sentence: '"Sacrifice": The author illustrates the personal sacrifices made
    by the directors, emphasizing their dedication and the unsustainable nature of
    their commitment.'
  sentences:
  - '"Emotional Appeal": This sentence evokes empathy by referencing painful experiences
    of fathers, aiming to resonate with readers'' emotions and highlight the importance
    of positive father figures.'
  - '"Value Proposition": The author emphasizes the altruistic motives of the directors,
    positioning them as committed to social impact rather than profit.'
  - '"Conditional Statement": This sentence highlights the necessity of strong leadership
    and management for military effectiveness, suggesting that without these elements,
    progress against Boko Haram is unattainable.'
- source_sentence: '"Future Projection": This sentence projects into the future, creating
    a sense of anticipation and framing the discussion around the transition of presidential
    power.'
  sentences:
  - '"Critique": The statement critiques the current state of negotiations, emphasizing
    the stagnation and failure to reach an agreement, which reflects poorly on the
    Conservatives.'
  - '"Dismissive Rhetoric": The mayor''s statement dismisses the legitimacy of the
    complaint by questioning the existence of the complainant, reinforcing the idea
    that the majority should not be inconvenienced by a single dissenting voice.'
  - '"Future Action": This sentence anticipates upcoming legislative proposals, reinforcing
    the urgency and seriousness of the issue while highlighting key political figures
    involved.'
- source_sentence: '"Provocative Question": The rhetorical question challenges the
    reader to consider the lack of dignity in labor, prompting reflection on the nature
    of their work.'
  sentences:
  - '"Cautionary Reflection": The author expresses concern about the misuse of the
    concept of freedom, suggesting that it can lead to moral irresponsibility.'
  - '"Clarification of Intent": This sentence clarifies that detachment is not about
    anger but about wishing well for others, reinforcing the positive nature of the
    practice.'
  - '"Engagement with Skepticism": This sentence directly addresses potential skepticism
    from the reader, inviting them to reconsider their views on the topic.'
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
      value: 0.6119042038917542
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
    '"Provocative Question": The rhetorical question challenges the reader to consider the lack of dignity in labor, prompting reflection on the nature of their work.',
    '"Clarification of Intent": This sentence clarifies that detachment is not about anger but about wishing well for others, reinforcing the positive nature of the practice.',
    '"Cautionary Reflection": The author expresses concern about the misuse of the concept of freedom, suggesting that it can lead to moral irresponsibility.',
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

* Dataset: `all-nli-dev`
* Evaluated with [<code>TripletEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.TripletEvaluator)

| Metric              | Value      |
|:--------------------|:-----------|
| **cosine_accuracy** | **0.6119** |

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
* Size: 150,000 training samples
* Columns: <code>anchor</code>, <code>positive</code>, and <code>negative</code>
* Approximate statistics based on the first 1000 samples:
  |         | anchor                                                                             | positive                                                                           | negative                                                                           |
  |:--------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
  | type    | string                                                                             | string                                                                             | string                                                                             |
  | details | <ul><li>min: 24 tokens</li><li>mean: 38.83 tokens</li><li>max: 62 tokens</li></ul> | <ul><li>min: 24 tokens</li><li>mean: 38.17 tokens</li><li>max: 60 tokens</li></ul> | <ul><li>min: 23 tokens</li><li>mean: 38.63 tokens</li><li>max: 62 tokens</li></ul> |
* Samples:
  | anchor                                                                                                                                                                                                      | positive                                                                                                                                                                                                                                                                 | negative                                                                                                                                                                                                                                                          |
  |:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>"Call for Support": The author calls for increased support and backing for mental health organizations, urging the reader to consider the importance of collective action.</code>                     | <code>"Progress Report": This sentence indicates ongoing efforts to rebuild the army, suggesting a positive development amidst previous failures, and aims to instill a sense of hope for recovery.</code>                                                               | <code>"Corporate Messaging": Highlights the positive spin Coca-Cola puts on its sponsorship, framing it as beneficial to public happiness.</code>                                                                                                                 |
  | <code>"Statistical Evidence": This sentence uses specific statistics to illustrate the extent of the problem, lending credibility to the argument by quantifying the number of individuals involved.</code> | <code>"PR Perspective": This statement emphasizes the public relations aspect of the bank's actions, suggesting that the author is critiquing the potential image consequences of the loan, thus appealing to the reader's concern for reputation and perception.</code> | <code>"Recognition of Impact": The author underscores the significance of the error by linking it to Clooney's public profile, suggesting that her status amplifies the responsibility of the publication.</code>                                                 |
  | <code>"Identify Avoidance": This sentence emphasizes the tendency to avoid difficult questions, reinforcing the argument that such avoidance is detrimental to addressing mental health issues.</code>      | <code>"Victimization and Neglect": This sentence emphasizes the systemic neglect and mistreatment of victims by university administrations, framing their public disclosures as a necessary response to being ignored.</code>                                            | <code>"Causal Inquiry": This rhetorical question implies that the focus of surveillance is misplaced, suggesting that the government is more concerned with monitoring its citizens than addressing real threats, thus critiquing governmental priorities.</code> |
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
* Size: 38,457 evaluation samples
* Columns: <code>anchor</code>, <code>positive</code>, and <code>negative</code>
* Approximate statistics based on the first 1000 samples:
  |         | anchor                                                                             | positive                                                                           | negative                                                                           |
  |:--------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
  | type    | string                                                                             | string                                                                             | string                                                                             |
  | details | <ul><li>min: 24 tokens</li><li>mean: 38.04 tokens</li><li>max: 58 tokens</li></ul> | <ul><li>min: 24 tokens</li><li>mean: 38.36 tokens</li><li>max: 56 tokens</li></ul> | <ul><li>min: 24 tokens</li><li>mean: 38.55 tokens</li><li>max: 62 tokens</li></ul> |
* Samples:
  | anchor                                                                                                                                                                                                                           | positive                                                                                                                                                                                           | negative                                                                                                                                                                                                        |
  |:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>"Specific Example": By citing the Ebola crisis, the author provides a concrete example of the broader issues of conflict and health, illustrating the real-world implications of inaction.</code>                          | <code>"Emotional Appeal": This sentence evokes a sense of tragedy and fear regarding the burial of the victims, highlighting the cultural and emotional stakes involved.</code>                    | <code>"Call to Action": This statement advocates for the need for more individuals like Rice to combat bullying, framing the issue as a societal responsibility.</code>                                         |
  | <code>"Economic Assessment": This sentence summarizes the overall economic condition of Iran, indicating that it is insufficient to support its regional ambitions, thereby framing the economic context of the argument.</code> | <code>"Proposition": The author suggests a potential solution for the Afghan government to improve its situation by reducing costs, indicating a proactive approach to governance.</code>          | <code>"Critique": This sentence critiques the government's current approach by listing its failures and inconsistencies, highlighting a lack of coherent policy direction.</code>                               |
  | <code>"Shock Value": The author uses the stark statement about the death penalty to evoke a strong emotional response and highlight the extreme nature of the law.</code>                                                        | <code>"Exaggeration for Effect": The sentence uses hyperbolic language to emphasize the vastness of the Muslim faith, which could be interpreted as an attempt to provoke concern or alarm.</code> | <code>"Defense of Character": This statement serves to defend Cochran's character by asserting that he did not engage in discriminatory behavior, reinforcing his innocence in the eyes of the audience.</code> |
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
- `per_device_train_batch_size`: 32
- `learning_rate`: 2e-05
- `num_train_epochs`: 1
- `warmup_ratio`: 0.1
- `batch_sampler`: no_duplicates

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: epoch
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 32
- `per_device_eval_batch_size`: 8
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
- `num_train_epochs`: 1
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
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
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
- `ddp_find_unused_parameters`: None
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
| Epoch  | Step | Training Loss | Validation Loss | all-nli-dev_cosine_accuracy |
|:------:|:----:|:-------------:|:---------------:|:---------------------------:|
| -1     | -1   | -             | -               | 0.5010                      |
| 0.0213 | 100  | 4.2723        | -               | -                           |
| 0.0427 | 200  | 3.9299        | -               | -                           |
| 0.0640 | 300  | 3.8555        | -               | -                           |
| 0.0853 | 400  | 3.8642        | -               | -                           |
| 0.1067 | 500  | 3.8591        | -               | -                           |
| 0.1280 | 600  | 3.833         | -               | -                           |
| 0.1493 | 700  | 3.8416        | -               | -                           |
| 0.1706 | 800  | 3.8183        | -               | -                           |
| 0.1920 | 900  | 3.8193        | -               | -                           |
| 0.2133 | 1000 | 3.7987        | -               | -                           |
| 0.2346 | 1100 | 3.8138        | -               | -                           |
| 0.2560 | 1200 | 3.7821        | -               | -                           |
| 0.2773 | 1300 | 3.7797        | -               | -                           |
| 0.2986 | 1400 | 3.7894        | -               | -                           |
| 0.3200 | 1500 | 3.7825        | -               | -                           |
| 0.3413 | 1600 | 3.7534        | -               | -                           |
| 0.3626 | 1700 | 3.7656        | -               | -                           |
| 0.3840 | 1800 | 3.744         | -               | -                           |
| 0.4053 | 1900 | 3.746         | -               | -                           |
| 0.4266 | 2000 | 3.7293        | -               | -                           |
| 0.4480 | 2100 | 3.718         | -               | -                           |
| 0.4693 | 2200 | 3.7099        | -               | -                           |
| 0.4906 | 2300 | 3.7288        | -               | -                           |
| 0.5119 | 2400 | 3.7302        | -               | -                           |
| 0.5333 | 2500 | 3.696         | -               | -                           |
| 0.5546 | 2600 | 3.6941        | -               | -                           |
| 0.5759 | 2700 | 3.6801        | -               | -                           |
| 0.5973 | 2800 | 3.7125        | -               | -                           |
| 0.6186 | 2900 | 3.7023        | -               | -                           |
| 0.6399 | 3000 | 3.7078        | -               | -                           |
| 0.6613 | 3100 | 3.6839        | -               | -                           |
| 0.6826 | 3200 | 3.6588        | -               | -                           |
| 0.7039 | 3300 | 3.6835        | -               | -                           |
| 0.7253 | 3400 | 3.6759        | -               | -                           |
| 0.7466 | 3500 | 3.6481        | -               | -                           |
| 0.7679 | 3600 | 3.6851        | -               | -                           |
| 0.7892 | 3700 | 3.6926        | -               | -                           |
| 0.8106 | 3800 | 3.6301        | -               | -                           |
| 0.8319 | 3900 | 3.6638        | -               | -                           |
| 0.8532 | 4000 | 3.653         | -               | -                           |
| 0.8746 | 4100 | 3.6625        | -               | -                           |
| 0.8959 | 4200 | 3.6742        | -               | -                           |
| 0.9172 | 4300 | 3.6649        | -               | -                           |
| 0.9386 | 4400 | 3.6671        | -               | -                           |
| 0.9599 | 4500 | 3.6676        | -               | -                           |
| 0.9812 | 4600 | 3.6307        | -               | -                           |
| 1.0    | 4688 | -             | 2.2697          | 0.6119                      |


### Framework Versions
- Python: 3.12.2
- Sentence Transformers: 4.1.0
- Transformers: 4.51.3
- PyTorch: 2.6.0
- Accelerate: 1.0.0
- Datasets: 2.18.0
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