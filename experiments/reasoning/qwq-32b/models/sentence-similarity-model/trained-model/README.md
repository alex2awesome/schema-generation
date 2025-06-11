---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:12717
- loss:MultipleNegativesRankingLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: '"Pattern recognition": This step recognizes a pattern in the representation
    of complex numbers in polar form, specifically that the modulus of the product
    is the product of the moduli, and applies this pattern to simplify the calculation.'
  sentences:
  - "\"Symbolic Manipulation\": This step involves manipulating algebraic equations\
    \ to solve for variables and simplify expressions. The reasoning focuses on applying\
    \ algebraic rules and properties to transform the equations and arrive at a solution.\
    \ \n\n\n"
  - '"Expansion": This step performs the expansion of the binomial expressions (u
    + 0.5)^3 and (u - 0.5)^3. The keyword "Expansion" is chosen because the reasoning
    involves breaking down the binomials into their polynomial components, which is
    a fundamental algebraic operation. Your response:

    "Expansion": This step performs the expansion of the binomial expressions (u +
    0.5)^3 and (u - 0.5)^3. The keyword "Expansion" is chosen because the reasoning
    involves breaking down the binomials into their poly'
  - '"Formula Application": This step involves recognizing a mathematical formula
    that applies to the problem at hand, and applying it to the specific case, by
    substituting the relevant values into the formula and preparing to perform the
    necessary calculations to arrive at the solution.'
- source_sentence: "\"Substitution\": This step substitutes the previously defined\
    \ expressions for *x* and *y* into the equation *xy = 64*. This is a direct application\
    \ of the given information to simplify the problem. \n\n\n"
  sentences:
  - "\"Deduction\": This step concludes that the highest possible value for N is 34\
    \ based on previous reasoning that showed N=34 is achievable and N=35 is impossible.\
    \ \n\n\n"
  - "\"Symbolic Manipulation\": This step primarily involves manipulating mathematical\
    \ symbols and equations to arrive at a solution. It uses substitution, algebraic\
    \ operations (cross-multiplication, expansion, simplification), and the quadratic\
    \ formula to transform the given relationship into a solvable equation. \n\n\n"
  - '"algebraic manipulation": Solving for g(-1) by isolating it on one side of the
    equation through basic algebraic operations. To get g(-1) alone, subtract 3 from
    both sides and then divide by -2. This step uses straightforward algebra to find
    the value of the function g at -1. To solve for g(-1), the equation (-2)g(-1)
    + 3 = 1 is manipulated algebraically. First, 3 is subtracted from both sides to
    get (-2)g(-1) = -2. Then, both sides are divided by -2 to isolate g(-1), resulting
    in g(-1) = 1. This'
- source_sentence: "\"Calculation\": This step performs a series of arithmetic calculations\
    \ to combine the volume contributions from different parts of the expanded shape\
    \ and express the final result in the desired form (m + nπ)/p. \n\n\n"
  sentences:
  - "\"Algebraic Manipulation\": This step performs basic algebraic operations (subtraction,\
    \ combining like terms) to simplify the expression. \n\n\n"
  - "\"Substitution\": This step identifies a known relationship (sin θ = r/a) and\
    \ decides to replace all instances of sin θ with r/a in the subsequent calculations.\
    \ This is a direct application of substitution to simplify the expressions. \n\
    \n\n"
  - "\"Pattern Recognition\": The reasoning identifies a trend in the examples provided\
    \ (each added rectangle potentially adds one region) and generalizes this trend\
    \ to conclude that the maximum number of regions is n+1. \n\n\n"
- source_sentence: '"algebraic manipulation": Simplifying the expression by combining
    like terms and applying basic algebraic rules. To get from 2(n+1)(n(n+1)/2) to
    n(n+1)^2, we distribute and simplify the terms. The step is purely algebraic,
    focusing on rearranging and simplifying the equation to make it more manageable
    for further analysis. To ensure the step is clear, we can break it down: 2(n+1)(n(n+1)/2)
    = (n+1)(n(n+1)) = n(n+1)^2. This simplification is crucial for the subsequent
    inequality comparison. To ver'
  sentences:
  - '"algebraic manipulation": Simplifying the expression for the area of the octagon
    by combining like terms and factoring out common factors to reach a more compact
    and understandable form. To ensure the correctness of the algebraic steps, the
    expression is simplified step-by-step, leading to the final form \(2a^2(\sqrt{2}
    - 1)\). To verify the result, the simplified expression is compared with the original
    form to ensure they are equivalent. To further validate, a numerical example is
    used to chec'
  - "\"Problem Decomposition\": This step breaks down the complex problem into smaller,\
    \ more manageable subproblems. It identifies the key decision point (A's choice\
    \ of the first opponent) and outlines the possible scenarios that arise based\
    \ on that decision. \n\n\n"
  - '"Derivation": This step performs a derivation, which is a process of generating
    a new statement from a given statement, in this case, the first inequality, by
    applying a set of rules or transformations, in this case, the rules of algebra.
    The goal of the derivation is to transform the original statement into a new statement
    that is more useful or easier to work with, in this case, to find the range of
    values for k. The derivation is "similar" to the previous one, indicating that
    the same set of '
- source_sentence: "\"Simplification\": This step combines like terms to reduce the\
    \ complexity of the expression. \n\n\n"
  sentences:
  - '"Verification": This step performs verification by checking if the sum of two
    sides of the triangle is less than or equal to the third side, which is a necessary
    condition for the triangle inequality to be violated, and thus for the three pieces
    to no longer form a triangle.'
  - "\"Deduction\": This step uses previously established facts (p divides 2^q -1\
    \ and q divides 2^p -1) and logical rules to arrive at a new conclusion (q divides\
    \ p-1 and p divides q-1). \n\n\n"
  - '"Index Manipulation": This step involves changing the starting point of summations
    to align the powers of x, which is a common technique in manipulating series expansions
    to facilitate combining like terms.'
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
      value: 0.7496463656425476
      name: Cosine Accuracy
  - task:
      type: triplet
      name: Triplet
    dataset:
      name: all nli dev trainer
      type: all-nli-dev-trainer
    metrics:
    - type: cosine_accuracy
      value: 0.893210768699646
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
    '"Simplification": This step combines like terms to reduce the complexity of the expression. \n\n\n',
    '"Index Manipulation": This step involves changing the starting point of summations to align the powers of x, which is a common technique in manipulating series expansions to facilitate combining like terms.',
    '"Deduction": This step uses previously established facts (p divides 2^q -1 and q divides 2^p -1) and logical rules to arrive at a new conclusion (q divides p-1 and p divides q-1). \n\n\n',
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
| **cosine_accuracy** | **0.7496**  | **0.8932**          |

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
* Size: 12,717 training samples
* Columns: <code>anchor</code>, <code>positive</code>, and <code>negative</code>
* Approximate statistics based on the first 1000 samples:
  |         | anchor                                                                              | positive                                                                            | negative                                                                            |
  |:--------|:------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|
  | type    | string                                                                              | string                                                                              | string                                                                              |
  | details | <ul><li>min: 21 tokens</li><li>mean: 69.91 tokens</li><li>max: 159 tokens</li></ul> | <ul><li>min: 18 tokens</li><li>mean: 69.94 tokens</li><li>max: 170 tokens</li></ul> | <ul><li>min: 17 tokens</li><li>mean: 67.87 tokens</li><li>max: 158 tokens</li></ul> |
* Samples:
  | anchor                                                                                                                                                                                                                                                                                                                                       | positive                                                                                                                                                                                                                                                                                              | negative                                                                                                                                                                                                                                                                                                                                                       |
  |:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>"Multiplication": This step performs multiplication of two numbers, 2015 and 3, and then adds the result to another number, 141,050, to compute the final product of 2015 and 73. The reasoning is focused on executing a specific arithmetic operation to obtain a partial result that contributes to the overall calculation.</code> | <code>"Arithmetic Computation": This step performs a basic arithmetic operation of multiplication, where the result of multiplying 2 by 2.25 is computed to obtain 4.5. The reasoning here is straightforward and involves applying a basic mathematical rule to arrive at a numerical answer.</code> | <code>"Division": This step performs a division operation, breaking down the dividend into multiples of the divisor and a remainder, and then combining the results to obtain the quotient.</code>                                                                                                                                                             |
  | <code>"Conclusion": This step performs the abstract level of reasoning of drawing a conclusion based on the evidence and reasoning presented in the previous steps, where the conclusion is a definitive answer to the problem.</code>                                                                                                       | <code>"Deduction": This step uses given information (parallel lines, cyclic quadrilateral properties) to logically deduce new facts (DF=BC, DFCB is a parallelogram). <br><br><br></code>                                                                                                             | <code>"Inductive Hypothesis": This step performs inductive reasoning by observing a pattern in the specific cases (n=5 and n=6) and hypothesizing that the pattern holds in general, i.e., the gcd is always 1. The reasoning is based on the assumption that the observed pattern will continue to hold for all n, without providing a rigorous proof.</code> |
  | <code>"Expression Simplification": This step performs expression simplification, where the individual terms of an algebraic expression are combined and rearranged to obtain a simpler form, in this case, combining like terms to simplify the left-hand side of the equation.</code>                                                       | <code>"Equation Manipulation": This step performs algebraic manipulations to solve for x in terms of y and then further manipulates the resulting expression to match the form of the function f(t) = t/(1 - t), ultimately leading to the conclusion that x can be expressed as -f(-y).</code>       | <code>"Applying Formula": This step applies a previously established formula for calculating the number of divisors of a number, given its prime factorization. The formula is used to derive the number of divisors from the exponents of the prime factors.</code>                                                                                           |
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
* Size: 1,414 evaluation samples
* Columns: <code>anchor</code>, <code>positive</code>, and <code>negative</code>
* Approximate statistics based on the first 1000 samples:
  |         | anchor                                                                              | positive                                                                            | negative                                                                            |
  |:--------|:------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|
  | type    | string                                                                              | string                                                                              | string                                                                              |
  | details | <ul><li>min: 19 tokens</li><li>mean: 68.13 tokens</li><li>max: 163 tokens</li></ul> | <ul><li>min: 19 tokens</li><li>mean: 68.54 tokens</li><li>max: 160 tokens</li></ul> | <ul><li>min: 20 tokens</li><li>mean: 68.03 tokens</li><li>max: 165 tokens</li></ul> |
* Samples:
  | anchor                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | positive                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | negative                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
  |:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>"reformulation": The step involves considering a different way to express the equation, aiming to simplify or make it more manageable. The keyword "reformulation" captures the essence of this process, as it reflects the attempt to transform the equation into a potentially more useful form. To analyze the reasoning step in question:<br><br>"reformulation": The step involves considering a different way to express the equation, aiming to simplify or make it more manageable. The keyword "reformulation" ca</code> | <code>"Alternative forms": This step considers the possibility of expressing the integral in different forms, particularly using other trigonometric identities, but concludes that the form involving arctan and tan is standard and appropriate for the given problem. The reasoning acknowledges the flexibility in representing trigonometric integrals but emphasizes the standard form as the most suitable answer. To ensure the solution is comprehensive, it briefly touches on the periodic nature of trigonomet</code>       | <code>"Interleaving rows": This step explores the idea of interleaving parts of rows in the original order and parts in reverse order to potentially increase the number of fixed points. The reasoning considers the complexity this introduces in maintaining valid transitions between rows and the challenge of connecting rows that are partially in their original order. The step also reflects on the difficulty of achieving more than 50 fixed points by this method, as it would require careful planning to av</code>       |
  | <code>"Parity Analysis": This step performs a parity analysis, examining the implications of having an odd number of vertices in a graph with certain properties, and using this to deduce the existence of a cycle of odd length. The reasoning relies on the fact that the sum of even numbers is even, and the presence of an odd number of vertices implies that there must be at least one odd-length cycle.</code>                                                                                                                | <code>"pattern recognition": The step involves recognizing and repeating a pattern of numbers that do not sum to 3. This is a form of pattern recognition where the individual identifies a sequence that meets the criteria and continues to use it to avoid sums of 3. To ensure the response is formatted as requested:<br><br>"pattern recognition": The step involves recognizing and repeating a pattern of numbers that do not sum to 3. This is a form of pattern recognition where the individual identifies a sequence</code> | <code>"case analysis": This step involves considering different scenarios or cases to explore the possible configurations of the triangle ADC. The reasoning here is to systematically evaluate the outcomes of drawing the perpendicular AD in two different directions (upwards and downwards) to determine the properties of the resulting obtuse triangle. This approach ensures that all potential solutions are considered and helps in identifying the correct configuration.<br>Your response:<br>"case analysis": This </code> |
  | <code>"Verification": This step performs verification of the derived inverse function by testing it with the original function and checking if the compositions yield the expected results, and also by cross-validating with a standard formula for linear functions and testing with specific values.</code>                                                                                                                                                                                                                          | <code>"Confirmation": This step performs a confirmation of the correctness of the equivalent point in standard spherical coordinates, where the reasoner is checking if the derived point meets the required conditions and ranges for the standard spherical coordinates, and is satisfied that it does, before moving on to further verification.</code>                                                                                                                                                                              | <code>"Recapitulation": This step performs a high-level review of the entire reasoning process, summarizing the key steps and conclusions drawn, to ensure that no critical details were overlooked and that the solution is comprehensive and accurate.</code>                                                                                                                                                                                                                                                                         |
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
- `per_device_eval_batch_size`: 32
- `gradient_accumulation_steps`: 2
- `learning_rate`: 2e-05
- `num_train_epochs`: 2
- `warmup_ratio`: 0.1
- `batch_sampler`: no_duplicates

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: epoch
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 32
- `per_device_eval_batch_size`: 32
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 2
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
| Epoch  | Step | Training Loss | Validation Loss | all-nli-dev_cosine_accuracy | all-nli-dev-trainer_cosine_accuracy |
|:------:|:----:|:-------------:|:---------------:|:---------------------------:|:-----------------------------------:|
| -1     | -1   | -             | -               | 0.7496                      | -                                   |
| 0.5025 | 100  | 7.1449        | -               | -                           | -                                   |
| 1.0    | 199  | -             | 2.9852          | -                           | 0.8911                              |
| 1.0050 | 200  | 6.1646        | -               | -                           | -                                   |
| 1.5075 | 300  | 5.9622        | -               | -                           | -                                   |
| 2.0    | 398  | -             | 2.9430          | -                           | 0.8932                              |


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