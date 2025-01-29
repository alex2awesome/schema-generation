# schema-generation

In this project, we aim to build techniques for automatically inferring schemas for arbitrary questions. Our method for constructing schemas proceeds in three parts:

![Schema Generation](assets/schema-learning-diagram.png)

* __Step 1__: We generate a superset of schema tags by prompting an LLM (gray box) to label each input paragraph (orange boxes) with a keyword tag label (yellow boxes). We are testing to see how well this works for arbitrary schemas: so far, we have tested this with narrative-role schemas. In early experiments, we have found that schema superset generation should ideally be done without reference to any few-shot examples or guidance whatsoever, in order to get an unbiased and large superset.
* __Step 2__: We train a clustering algorithm to cluster similar elements in the schema superset together. In order to do this, we sample a small amount of similar pairwise examples, and prompt an LLM to identify if they are the same or not. Then, we use these silver-labels to fine-tune an SBERT embedding model. Next, we use this fine-tuned SBERT model to embed all of our labeled documents. Finally, we run UMAP+HDBSCAN on the fine-tuned embeddings to discover clusters.
* __Step 3__: Optionally, we test the efficacy of the schema. This can be useful for tuning hyperparameters. We use a method introduced in \[1\], called _latent perplexity_. In _latent perplexity_, an autoregressive GPT2 model is trained to take the schema elements of all paragraphs, and assess the likelihood of the observed text (this was tested for a scenario where all paragraphs in a document were labeled, alternatively, we can also test with a single document label to predict the entire document).

\[1\] Spangher, Alexander, et al. "Explaining Mixtures of Sources in News Articles." Findings of the Association for Computational Linguistics: EMNLP 2024. 2024.