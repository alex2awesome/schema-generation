---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:20413
- loss:MultipleNegativesRankingLoss
base_model: microsoft/mpnet-base
widget:
- source_sentence: '"Algebraic manipulation": This step performs algebraic manipulation,
    where the original expression for volume V is transformed through a series of
    algebraic operations, such as multiplication, division, and rearrangement of terms,
    to arrive at a simplified expression. The goal is to reorganize the terms to facilitate
    further analysis or to reveal a pattern, but the underlying mathematical relationships
    and assumptions remain unchanged.'
  sentences:
  - "\"Substitution\": This step substitutes a previously defined expression for 'D'\
    \ with a new expression that is derived from it. \n\n\n"
  - '"substitution": Replacing M with its expression in terms of a and r, then raising
    it to the power of (n - l). To show the transformation of M into a form that can
    be multiplied with the other terms.

    Your response:

    "substitution": Replacing M with its expression in terms of a and r, then raising
    it to the power of (n - l). This step transforms M into a form that can be combined
    with the other terms in the product.'
  - "\"**Position Calculation**\": This step calculates the position of Boat 1 at\
    \ a given time *t* based on its speed and the starting point. It uses the formula:\
    \ position = speed * time. \n\n\n"
- source_sentence: '"Visualization": This step involves creating a mental image of
    the problem, recalling the geometric configuration, and setting the stage for
    further analysis. The reasoning is focused on understanding the spatial relationships
    and layout of the given elements, such as the circle, tangents, and triangle,
    to facilitate a deeper exploration of the problem.'
  sentences:
  - "\"Deduction\": This step arrives at a specific conclusion (\"Number with exactly\
    \ two 'HMMT's is 1\") based on a clear understanding of the problem constraints.\
    \ The reasoning is direct and follows logically from the given information. \n\
    \n\n"
  - '"Classification": This step performs a classification task, where the number
    6 is being evaluated against a set of criteria (being a perfect square) and is
    being assigned a label (not a perfect square) based on that evaluation.'
  - '"Pattern recognition": This step involves recognizing a pattern in the geometric
    configuration of the cube, specifically that the intersection of three faces (A,
    B, C) is a single point, the corner cube. This requires the ability to visualize
    and understand the spatial relationships between the different parts of the cube.'
- source_sentence: "\"Variable Substitution\": This step sets up a relationship between\
    \ the variables x and y based on the total number of stamps being 6. It's essentially\
    \ saying that the sum of the number of 3-cent stamps (x) and 4-cent stamps (y)\
    \ must equal 6. \n\n\n"
  sentences:
  - "\"Algebraic Manipulation\": This step involves rearranging the terms of an equation\
    \ to isolate a specific variable. It's a fundamental operation in algebra, aiming\
    \ to transform the equation without changing its overall meaning. \n\n\n"
  - '"Algebraic Simplification": This step involves breaking down a complex algebraic
    expression into its constituent parts, expanding and combining like terms, and
    reassembling the expression into a simpler form. The goal is to transform the
    original expression into a more manageable and understandable format, which can
    facilitate further analysis and problem-solving.'
  - "\"Algebraic Manipulation\": This step performs a series of algebraic operations\
    \ to simplify an expression. It involves distributing, combining like terms, and\
    \ factoring to rewrite the original expression in a more compact form. \n\n\n"
- source_sentence: '"Verification": This step performs verification by double-checking
    the conclusion drawn from previous steps, specifically to ensure that no palindromic
    numbers were missed between 101 and 131, thereby confirming the correctness of
    the conclusion that 131 is the second-smallest three-digit palindromic prime.'
  sentences:
  - '"Equation manipulation": This step involves algebraic manipulation of the equation
    to simplify it, specifically by combining like terms and performing arithmetic
    operations to isolate the trigonometric terms.'
  - '"Verification and Exploration": This step involves verifying the initial results
    by re-examining the data and exploring alternative explanations or methods to
    confirm the findings, demonstrating a deeper understanding of the problem and
    its solution.'
  - '"Clarification": This step performs clarification by explicitly distinguishing
    between two possible interpretations of the problem, ensuring that the correct
    interpretation (reciprocal of the sum) is being used, and acknowledging the importance
    of the parentheses in disambiguating the expression.'
- source_sentence: '"Generalization": The step extends the previous reasoning from
    a specific case (n=2) to a more general case (n=3). This involves applying the
    same method of counting non-negative integer solutions to a new, slightly more
    complex scenario to verify the consistency and correctness of the approach. To
    ensure the method works for different values of n, the step demonstrates the process
    with a new example, thereby generalizing the initial reasoning. To further validate
    the method, the step lists all '
  sentences:
  - '"Analogy": This step performs an analogy by recognizing a pattern in the previous
    case (minute 2 was +y) and applying it to a new, but similar, case (minute 2 was
    -y), assuming the outcome will be similar.'
  - '"Verification": This step performs verification by testing the hypothesis that
    the product of the three areas is the square of the volume with a specific example,
    and confirming that the result holds true, thereby increasing confidence in the
    conclusion.'
  - "\"Case Analysis\": This step begins to explore a specific instance within a set\
    \ of possible solutions. It selects the first value from the set of possible values\
    \ for 'x' and proceeds to analyze the consequences of that choice. \n\n\n"
pipeline_tag: sentence-similarity
library_name: sentence-transformers
metrics:
- cosine_accuracy
model-index:
- name: SentenceTransformer based on microsoft/mpnet-base
  results:
  - task:
      type: triplet
      name: Triplet
    dataset:
      name: all nli dev
      type: all-nli-dev
    metrics:
    - type: cosine_accuracy
      value: 0.6317335963249207
      name: Cosine Accuracy
---

# SentenceTransformer based on microsoft/mpnet-base

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [microsoft/mpnet-base](https://huggingface.co/microsoft/mpnet-base) on the json dataset. It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [microsoft/mpnet-base](https://huggingface.co/microsoft/mpnet-base) <!-- at revision 6996ce1e91bd2a9c7d7f61daec37463394f73f09 -->
- **Maximum Sequence Length:** 512 tokens
- **Output Dimensionality:** 768 dimensions
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
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': False}) with Transformer model: MPNetModel 
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
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
    '"Generalization": The step extends the previous reasoning from a specific case (n=2) to a more general case (n=3). This involves applying the same method of counting non-negative integer solutions to a new, slightly more complex scenario to verify the consistency and correctness of the approach. To ensure the method works for different values of n, the step demonstrates the process with a new example, thereby generalizing the initial reasoning. To further validate the method, the step lists all ',
    '"Analogy": This step performs an analogy by recognizing a pattern in the previous case (minute 2 was +y) and applying it to a new, but similar, case (minute 2 was -y), assuming the outcome will be similar.',
    '"Case Analysis": This step begins to explore a specific instance within a set of possible solutions. It selects the first value from the set of possible values for \'x\' and proceeds to analyze the consequences of that choice. \n\n\n',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

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
| **cosine_accuracy** | **0.6317** |

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
* Size: 20,413 training samples
* Columns: <code>anchor</code>, <code>positive</code>, and <code>negative</code>
* Approximate statistics based on the first 1000 samples:
  |         | anchor                                                                             | positive                                                                            | negative                                                                            |
  |:--------|:-----------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|
  | type    | string                                                                             | string                                                                              | string                                                                              |
  | details | <ul><li>min: 21 tokens</li><li>mean: 67.8 tokens</li><li>max: 161 tokens</li></ul> | <ul><li>min: 18 tokens</li><li>mean: 68.35 tokens</li><li>max: 155 tokens</li></ul> | <ul><li>min: 15 tokens</li><li>mean: 67.46 tokens</li><li>max: 148 tokens</li></ul> |
* Samples:
  | anchor                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | positive                                                                                                                                                                                                                                                                                                                                      | negative                                                                                                                                                                                                                                                                                                                                                                                                        |
  |:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>"Conclusion": This step is the final conclusion of the reasoning process, summarizing the result of the calculations and confirming the answer to the problem. The keyword "Conclusion" captures the essence of this step, as it represents the end of the reasoning and the final answer to the problem. The description explains that this step is where the solution is finalized and presented. Your response:<br>"Conclusion": This step is the final conclusion of the reasoning process, summarizing the result </code> | <code>"Verification": This step involves checking the validity of a proposed solution (n=10 pieces) by manually calculating the sum of the lengths of the pieces to ensure it matches the total length of the wire (150 cm), thereby confirming the feasibility of the solution.</code>                                                       | <code>"Evaluation": This step performs an evaluation of the function y at a specific point x=π/2, by substituting the value of x into the function and computing the resulting value of y.</code>                                                                                                                                                                                                               |
  | <code>"Modular Arithmetic": This step performs modular arithmetic, specifically computing the residue of a number (5) modulo 3, which is a fundamental operation in number theory.</code>                                                                                                                                                                                                                                                                                                                                            | <code>"Division Check": This step performs a division check to see if the result of dividing 906 by 124 is an integer, which is a necessary condition for the amount to be achievable by the machine. The step is focused on verifying whether the division yields a whole number, which is a key aspect of the problem's constraints.</code> | <code>"Multiplication": This step performs a multiplication operation, but more abstractly, it is an example of applying a mathematical operation to combine two quantities, in this case, the result of the previous step (C2 / π) and the height (Sπ / C), to derive a new quantity, which is the volume of the cylinder. The focus is on the algebraic manipulation of the expression to simplify it.</code> |
  | <code>"Equation Restatement": This step takes an existing equation or condition and rephrases it in a more convenient or familiar form, in this case, expressing the sum of the x_i's in modular arithmetic notation.</code>                                                                                                                                                                                                                                                                                                         | <code>"Algebraic Manipulation": This step involves applying algebraic rules and properties to simplify the expression under the sixth root. It uses properties like the power of a product, the power of a power, and fractional exponents to rewrite the expression in a more manageable form. <br><br><br></code>                           | <code>"Reevaluation": This step involves reevaluating the approach taken so far, recognizing that the current method is becoming too cumbersome, and considering alternative perspectives or methods that might be more efficient or effective in solving the problem. The reasoning involves stepping back, reassessing the situation, and looking for a smarter or more insightful way to proceed.</code>     |
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
* Size: 20,413 evaluation samples
* Columns: <code>anchor</code>, <code>positive</code>, and <code>negative</code>
* Approximate statistics based on the first 1000 samples:
  |         | anchor                                                                              | positive                                                                            | negative                                                                            |
  |:--------|:------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|
  | type    | string                                                                              | string                                                                              | string                                                                              |
  | details | <ul><li>min: 19 tokens</li><li>mean: 67.41 tokens</li><li>max: 152 tokens</li></ul> | <ul><li>min: 18 tokens</li><li>mean: 69.42 tokens</li><li>max: 152 tokens</li></ul> | <ul><li>min: 17 tokens</li><li>mean: 68.84 tokens</li><li>max: 160 tokens</li></ul> |
* Samples:
  | anchor                                                                                                                                                                                                                                                                                       | positive                                                                                                                                                                                                                                                                                                                                 | negative                                                                                                                                                                                                                                                                                                                                                                                                             |
  |:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>"Symbolic Manipulation": This step involves a series of algebraic and trigonometric manipulations to simplify the expression for the distance QR and ultimately relate it to the sine function of the angle difference (a-b). <br><br><br></code>                                      | <code>"Algebraic Manipulation": This step involves a specific algebraic operation (squaring a fraction) to transform an expression into a form that is useful for completing the square, demonstrating a mechanical application of algebraic rules to achieve a specific goal.</code>                                                    | <code>"Self-reflection": The user is explicitly checking their own assumptions and reasoning process to ensure they are correctly interpreting the problem and applying the appropriate mathematical concepts. They are questioning whether their understanding of "combinations" versus "permutations" is accurate and if they have accounted for the order of selection in their calculations. <br><br><br></code> |
  | <code>"Verification": This step performs verification by checking the calculated answer against the provided options to ensure consistency and accuracy, and also involves a meta-level check of the problem statement to ensure that the correct option is indeed marked as correct.</code> | <code>"Error Detection": This step involves recognizing a potential mistake in the previous steps and deciding to re-examine the algebra to verify the correctness of the solution. The keyword "Error Detection" captures the essence of this step, which is to identify and address a potential error in the reasoning process.</code> | <code>"Conclusion": This step performs a conclusion, where the solution to the problem is explicitly stated, summarizing the results of the previous steps and providing the final answer.</code>                                                                                                                                                                                                                    |
  | <code>"Variable Substitution": This step sets up a relationship between the variables x and y based on the total number of stamps being 6. It's essentially saying that the sum of the number of 3-cent stamps (x) and 4-cent stamps (y) must equal 6. <br><br><br></code>                   | <code>"Algebraic Manipulation": This step involves rearranging the terms of an equation to isolate a specific variable. It's a fundamental operation in algebra, aiming to transform the equation without changing its overall meaning. <br><br><br></code>                                                                              | <code>"Algebraic Manipulation": This step performs a series of algebraic operations to simplify an expression. It involves distributing, combining like terms, and factoring to rewrite the original expression in a more compact form. <br><br><br></code>                                                                                                                                                          |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: steps
- `per_device_train_batch_size`: 64
- `per_device_eval_batch_size`: 64
- `learning_rate`: 2e-05
- `warmup_ratio`: 0.05
- `fp16`: True
- `batch_sampler`: no_duplicates

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: steps
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 64
- `per_device_eval_batch_size`: 64
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
- `warmup_ratio`: 0.05
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
- `fp16`: True
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
- `dispatch_batches`: None
- `split_batches`: None
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
| Epoch | Step | all-nli-dev_cosine_accuracy |
|:-----:|:----:|:---------------------------:|
| -1    | -1   | 0.6317                      |


### Framework Versions
- Python: 3.11.8
- Sentence Transformers: 3.4.1
- Transformers: 4.50.0.dev0
- PyTorch: 2.5.1+cu124
- Accelerate: 1.0.1
- Datasets: 3.0.2
- Tokenizers: 0.21.0

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