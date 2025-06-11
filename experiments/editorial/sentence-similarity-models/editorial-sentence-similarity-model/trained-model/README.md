---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:23153
- loss:MultipleNegativesRankingLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: '"Commitment Declaration": Affirms a serious commitment to personal
    change, appealing to the reader''s desire for accountability and determination.'
  sentences:
  - '"Affirmation": This sentence affirms the essential role of teens in the online
    ecosystem, reinforcing the idea that their presence is vital for the vibrancy
    and relevance of internet culture.'
  - '"Reference to Activism": This sentence introduces a prominent example of activism
    against campus sexual assault, highlighting the efforts of women to raise awareness
    and seek justice.'
  - '"Informative": The author provides factual information about Bush''s exploratory
    committee, which serves to inform the reader about the current status of his candidacy.'
- source_sentence: '"Illustration of Solidarity": This sentence illustrates a collective
    response to tragedy, emphasizing the unity and public mourning that emerged in
    reaction to the attacks.'
  sentences:
  - '"Thesis Statement": The author establishes a provocative claim that the root
    issue is not immigration but rather a collective amnesia regarding Europe''s past
    and its implications.'
  - '"Historical Context": This sentence establishes the historical precedent of attacks
    on Charlie Hebdo, reinforcing the idea that the publication has been a consistent
    target for violence.'
  - '"Evidence": The mention of escalating border militarization and deportations
    serves as further evidence of the administration''s controversial policies, underscoring
    the author''s argument about the need for a vocal anti-war movement.'
- source_sentence: '"Vision for Leadership": The author posits that true leadership
    involves finding common ground, advocating for a more unifying approach in political
    discourse.'
  sentences:
  - '"Value Proposition": This sentence articulates a value proposition, asserting
    that keeping London affordable and improving quality of life should be central
    to any future plans, appealing to the reader''s sense of fairness and community.'
  - '"Affirmation": The author asserts that the attack is indeed related to the broader
    racial crisis, reinforcing their argument against the previous denials and calling
    for recognition of systemic issues.'
  - '"Unity Statement": The volunteer''s declaration aims to promote solidarity among
    the group, framing their collective experience as a shared journey despite the
    unexpected circumstances.'
- source_sentence: '"Critique of Audience": This sentence critiques the right-wing
    audience''s shallow understanding of the film, suggesting that their reverence
    is misguided and highlights a broader issue of ideological appropriation.'
  sentences:
  - '"Suspense Building": The author builds suspense by indicating that more shocking
    information will follow, encouraging the reader to remain engaged and attentive.'
  - '"Misunderstanding of Political Dynamics": The author critiques the belief that
    messaging alone can resolve political issues, indicating a deeper misunderstanding
    of effective governance.'
  - '"Emotional Appeal": This sentence evokes strong emotions, highlighting the range
    of feelings experienced by the community, which aims to resonate with the reader''s
    empathy.'
- source_sentence: '"Clarification": The author emphasizes the distinction between
    collective expressions of solidarity and individual identities, suggesting that
    while people may empathize, they do not literally become the victims.'
  sentences:
  - '"Critique of Societal Attitudes": The author critiques societal reluctance to
    engage with mental health topics, arguing that this fear inhibits open dialogue
    about personal struggles.'
  - '"Historical Context": This sentence provides a factual assertion about the FBI''s
    complicity in racial violence, reinforcing the argument against the distortion
    of historical narratives.'
  - '"Denial and Threat": This sentence presents the North Korean regime''s denial
    of involvement while simultaneously highlighting their aggressive threats, illustrating
    the complexity of the situation and the regime''s contradictory behavior.'
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
      value: 0.9187718629837036
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
    '"Clarification": The author emphasizes the distinction between collective expressions of solidarity and individual identities, suggesting that while people may empathize, they do not literally become the victims.',
    '"Historical Context": This sentence provides a factual assertion about the FBI\'s complicity in racial violence, reinforcing the argument against the distortion of historical narratives.',
    '"Critique of Societal Attitudes": The author critiques societal reluctance to engage with mental health topics, arguing that this fear inhibits open dialogue about personal struggles.',
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
| **cosine_accuracy** | **0.9188** |

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
* Size: 23,153 training samples
* Columns: <code>anchor</code>, <code>positive</code>, and <code>negative</code>
* Approximate statistics based on the first 1000 samples:
  |         | anchor                                                                             | positive                                                                           | negative                                                                           |
  |:--------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
  | type    | string                                                                             | string                                                                             | string                                                                             |
  | details | <ul><li>min: 25 tokens</li><li>mean: 37.52 tokens</li><li>max: 57 tokens</li></ul> | <ul><li>min: 21 tokens</li><li>mean: 38.24 tokens</li><li>max: 56 tokens</li></ul> | <ul><li>min: 25 tokens</li><li>mean: 38.54 tokens</li><li>max: 56 tokens</li></ul> |
* Samples:
  | anchor                                                                                                                                                                                                                       | positive                                                                                                                                                                                                           | negative                                                                                                                                                                                                                                                        |
  |:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>"Historical Parallel": The author draws a historical parallel to underscore the potential consequences of fear in intellectual inquiry, suggesting that past critiques of religion faced similar backlash.</code>      | <code>"Characterization": The author highlights Mal's age and violent history to establish his credibility and allure within the group, suggesting that these traits are valued among the misfit teenagers.</code> | <code>"Historical Context": The author invokes historical precedent to argue that the establishment of military courts revives past conflicts between military and civilian authority, suggesting that this will complicate the fight against extremism.</code> |
  | <code>"Admission of Fallibility": The author acknowledges moments of carelessness, reinforcing the idea that even the most careful individuals can make mistakes.</code>                                                     | <code>"Historical Analogy": The author uses a historical analogy to illustrate the concept of innovation and failure, framing early adopters as visionaries despite setbacks.</code>                               | <code>"Example of Denial": The author uses a specific example of media reaction to illustrate denial and deflection regarding the moral implications of torture.</code>                                                                                         |
  | <code>"Counter-Argument": This sentence presents a counter-argument to Republican perspectives, asserting that California politicians can provide effective governance that combines regulation with economic growth.</code> | <code>"Historical Context": This sentence provides a timeline for the technological revolution, establishing a historical backdrop for the subsequent arguments about its impact on the economy.</code>            | <code>"Turning Point": This sentence marks a significant historical turning point, indicating the end of American dominance and setting up the narrative of loss and injustice that follows.</code>                                                             |
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
* Size: 2,573 evaluation samples
* Columns: <code>anchor</code>, <code>positive</code>, and <code>negative</code>
* Approximate statistics based on the first 1000 samples:
  |         | anchor                                                                             | positive                                                                          | negative                                                                           |
  |:--------|:-----------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
  | type    | string                                                                             | string                                                                            | string                                                                             |
  | details | <ul><li>min: 24 tokens</li><li>mean: 37.74 tokens</li><li>max: 58 tokens</li></ul> | <ul><li>min: 23 tokens</li><li>mean: 38.1 tokens</li><li>max: 61 tokens</li></ul> | <ul><li>min: 23 tokens</li><li>mean: 38.47 tokens</li><li>max: 59 tokens</li></ul> |
* Samples:
  | anchor                                                                                                                                                                                                                                        | positive                                                                                                                                                                                                                                               | negative                                                                                                                                                                                                                                     |
  |:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>"Assertion of Value": The author asserts that Muslims have a theoretical appreciation for the pen, suggesting a cultural and intellectual tradition that values written expression.</code>                                              | <code>"Challenge": The author challenges the effectiveness of the leaders' actions, questioning whether they will take meaningful steps against radical Islam.</code>                                                                                  | <code>"Highlighting Double Standards": This sentence points out the prevalence of unchallenged all-white casts in media, contrasting them with the scrutiny faced by proposals for diverse representation.</code>                            |
  | <code>"Pattern Recognition": This sentence identifies a pattern of repression by mentioning other media services that faced similar fates, which helps to contextualize the situation within a broader narrative of media suppression.</code> | <code>"Policy Shift": This sentence illustrates a significant policy shift under President Clinton, showcasing a move towards humanitarian engagement despite previous tensions, which serves to highlight the complexity of US-Cuba relations.</code> | <code>"Example": This sentence provides a historical example of a successful reparation model, lending credibility to the proposal and illustrating potential pathways for implementation.</code>                                            |
  | <code>"Denial of Reality": This sentence asserts that society is ignoring the plight of modern martyrs, framing it as a moral failing.</code>                                                                                                 | <code>"Media Critique": Here, the author critiques media coverage of gun deaths, suggesting that it misrepresents the issue, which serves to undermine opposing arguments.</code>                                                                      | <code>"Irony and Sarcasm": This sentence employs irony to emphasize the failure of the government to uphold its environmental commitments, suggesting that their claims of being the 'greenest government' are now seen as insincere.</code> |
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
- `learning_rate`: 2e-05
- `warmup_ratio`: 0.1
- `batch_sampler`: no_duplicates

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: epoch
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 8
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
- `num_train_epochs`: 3
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
| -1     | -1   | -             | -               | 0.5177                      |
| 0.0345 | 100  | 2.9163        | -               | -                           |
| 0.0691 | 200  | 2.8291        | -               | -                           |
| 0.1036 | 300  | 2.6315        | -               | -                           |
| 0.1382 | 400  | 2.5656        | -               | -                           |
| 0.1727 | 500  | 2.5115        | -               | -                           |
| 0.2073 | 600  | 2.4863        | -               | -                           |
| 0.2418 | 700  | 2.4583        | -               | -                           |
| 0.2763 | 800  | 2.4405        | -               | -                           |
| 0.3109 | 900  | 2.4277        | -               | -                           |
| 0.3454 | 1000 | 2.3708        | -               | -                           |
| 0.3800 | 1100 | 2.336         | -               | -                           |
| 0.4145 | 1200 | 2.2622        | -               | -                           |
| 0.4491 | 1300 | 2.2916        | -               | -                           |
| 0.4836 | 1400 | 2.2712        | -               | -                           |
| 0.5181 | 1500 | 2.2841        | -               | -                           |
| 0.5527 | 1600 | 2.224         | -               | -                           |
| 0.5872 | 1700 | 2.1519        | -               | -                           |
| 0.6218 | 1800 | 2.149         | -               | -                           |
| 0.6563 | 1900 | 2.0529        | -               | -                           |
| 0.6908 | 2000 | 2.0802        | -               | -                           |
| 0.7254 | 2100 | 2.0658        | -               | -                           |
| 0.7599 | 2200 | 2.0417        | -               | -                           |
| 0.7945 | 2300 | 1.9621        | -               | -                           |
| 0.8290 | 2400 | 1.9519        | -               | -                           |
| 0.8636 | 2500 | 1.8756        | -               | -                           |
| 0.8981 | 2600 | 1.9089        | -               | -                           |
| 0.9326 | 2700 | 1.8302        | -               | -                           |
| 0.9672 | 2800 | 1.7428        | -               | -                           |
| 1.0    | 2895 | -             | 1.5738          | 0.8193                      |
| 1.0017 | 2900 | 1.7497        | -               | -                           |
| 1.0363 | 3000 | 1.4628        | -               | -                           |
| 1.0708 | 3100 | 1.5464        | -               | -                           |
| 1.1054 | 3200 | 1.5185        | -               | -                           |
| 1.1399 | 3300 | 1.4329        | -               | -                           |
| 1.1744 | 3400 | 1.4485        | -               | -                           |
| 1.2090 | 3500 | 1.434         | -               | -                           |
| 1.2435 | 3600 | 1.3949        | -               | -                           |
| 1.2781 | 3700 | 1.343         | -               | -                           |
| 1.3126 | 3800 | 1.4262        | -               | -                           |
| 1.3472 | 3900 | 1.3577        | -               | -                           |
| 1.3817 | 4000 | 1.2377        | -               | -                           |
| 1.4162 | 4100 | 1.2817        | -               | -                           |
| 1.4508 | 4200 | 1.2513        | -               | -                           |
| 1.4853 | 4300 | 1.3344        | -               | -                           |
| 1.5199 | 4400 | 1.2461        | -               | -                           |
| 1.5544 | 4500 | 1.2654        | -               | -                           |
| 1.5889 | 4600 | 1.236         | -               | -                           |
| 1.6235 | 4700 | 1.2484        | -               | -                           |
| 1.6580 | 4800 | 1.2107        | -               | -                           |
| 1.6926 | 4900 | 1.1506        | -               | -                           |
| 1.7271 | 5000 | 1.2369        | -               | -                           |
| 1.7617 | 5100 | 1.1915        | -               | -                           |
| 1.7962 | 5200 | 1.0989        | -               | -                           |
| 1.8307 | 5300 | 1.0941        | -               | -                           |
| 1.8653 | 5400 | 1.068         | -               | -                           |
| 1.8998 | 5500 | 1.1606        | -               | -                           |
| 1.9344 | 5600 | 1.091         | -               | -                           |
| 1.9689 | 5700 | 1.1353        | -               | -                           |
| 2.0    | 5790 | -             | 0.9834          | 0.9083                      |
| 2.0035 | 5800 | 1.1104        | -               | -                           |
| 2.0380 | 5900 | 0.967         | -               | -                           |
| 2.0725 | 6000 | 0.8858        | -               | -                           |
| 2.1071 | 6100 | 0.9655        | -               | -                           |
| 2.1416 | 6200 | 0.8909        | -               | -                           |
| 2.1762 | 6300 | 0.8745        | -               | -                           |
| 2.2107 | 6400 | 0.9087        | -               | -                           |
| 2.2453 | 6500 | 0.9072        | -               | -                           |
| 2.2798 | 6600 | 0.8836        | -               | -                           |
| 2.3143 | 6700 | 0.8511        | -               | -                           |
| 2.3489 | 6800 | 0.8331        | -               | -                           |
| 2.3834 | 6900 | 0.8724        | -               | -                           |
| 2.4180 | 7000 | 0.8837        | -               | -                           |
| 2.4525 | 7100 | 0.7973        | -               | -                           |
| 2.4870 | 7200 | 0.8467        | -               | -                           |
| 2.5216 | 7300 | 0.8581        | -               | -                           |
| 2.5561 | 7400 | 0.8266        | -               | -                           |
| 2.5907 | 7500 | 0.8201        | -               | -                           |
| 2.6252 | 7600 | 0.8119        | -               | -                           |
| 2.6598 | 7700 | 0.7902        | -               | -                           |
| 2.6943 | 7800 | 0.8273        | -               | -                           |
| 2.7288 | 7900 | 0.9243        | -               | -                           |
| 2.7634 | 8000 | 0.8229        | -               | -                           |
| 2.7979 | 8100 | 0.8035        | -               | -                           |
| 2.8325 | 8200 | 0.7754        | -               | -                           |
| 2.8670 | 8300 | 0.805         | -               | -                           |
| 2.9016 | 8400 | 0.8479        | -               | -                           |
| 2.9361 | 8500 | 0.7722        | -               | -                           |
| 2.9706 | 8600 | 0.836         | -               | -                           |
| 3.0    | 8685 | -             | 0.8332          | 0.9188                      |


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