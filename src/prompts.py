# pip install -U git+https://github.com/nicholishen/tooldantic.git
from pydantic import BaseModel, create_model, field_validator, ValidationError
from typing import List, Optional, Literal
from annotated_types import Len
from typing import ClassVar, Annotated


GENERIC_SIMILARITY_PROMPT = """
  I will show you {k_i} pairs of of labels, all applied to different mathematical reasoning steps.

  Are the two labels in each pair describing similar reasoning steps in mathematical problem solving?
  Think broadly about what each label is describing. Don't pay attention to the specific step each label is describing.
  Answer with "Yes" or "No". Answer each in a JSON in the format:
  {{
    "pair_idx": PAIR_IDX,
    "label": 'Yes' or 'No'
  }}

  <example>
  1. Label 1: ```"Trial and Error": This step explores different ways to solve the problem, hoping to find a combination that simplifies it. It's a process of trying different options and discarding them when they don't lead to a simpler form.,``` Label 2: ```"Exploring Alternatives": This step performs a reasoning of exploring alternative ways to break down 2700 into factors, specifically considering the factorization 2700 = 9 * 300, and then evaluating if this leads to a different simplified form of the cube root.'```
  2. Label 1: ```"Quantitative Comparison": This step compares the cost after the interchange to the original cost, calculates the difference, and then expresses that difference as a percentage of the original cost to determine if it matches the 50% increase stated in the problem. ``` Label 2: ```"Evaluation": This step evaluates the outcome of a hypothetical scenario by comparing the result to the original, and determines that the change is not as required, thus eliminating a possible solution.'```
  3. Label 1: ```"Elimination": This step considers the centroid as a potential solution but quickly dismisses it.``` Label 2: ```"Meta-reflection": This step involves reflecting on the reasoning process itself, recognizing a past mistake, and understanding how the correct process leads to the correct solution.```
  4. Label 1: ```"Factoring": This step applies the factoring technique to simplify the equation, breaking it down into two expressions multiplied together. The goal is to identify potential solutions by recognizing that if the product of two factors is zero, at least one of the factors must be zero.``` Label 2: ```"Identifying": This step involves identifying the coefficients of the polynomial to be divided, which is a crucial step in setting up the synthetic division process. The reasoning here is focused on extracting and labeling the relevant information from the polynomial expression.```
  Answers:
    [
      {{
      "pair_idx": 1,
      "label": 'Yes'
      }},
      {{
        "pair_idx": 2,
        "label": 'Yes'
      }},
      {{
        "pair_idx": 3,
        "label": 'No'
      }},
      {{
        "pair_idx": 4,
        "label": 'No'
      }}
    ]
  </example>

  Now it's your turn:
  {samples_str}

  Answers:
"""

EDITORIAL_INITIAL_LABELING_PROMPT = """
  You will be given an editorial or opinion article as well as a sentence from that article. 

  For each sentence, give a generic keyword label to explain the role it plays in the author's overall argumentation strategy. 
  This means identifying the function of the sentence in the context of the broader persuasive structure — what purpose it serves in advancing, supporting, framing, or organizing the argument.
  Avoid summarizing the content or repeating the surface meaning. Your goal is to interpret each unit's intentional role in shaping the argumentative flow of the article.
  Think in terms of strategic purpose: how the piece is structured to influence belief, agreement, or perception over time.
  Focus on describing what each unit is doing rhetorically, Return in a JSON in the format: 
  
  {{
    "label": LABEL,
    "description": DESCRIPTION
  }}

  <article>
  {article}
  </article>

  <sentence>
  {sentence}
  </sentence>

  Your response:
"""


MULTI_SENTENCE_EDITORIAL_LABELING_PROMPT = """
You will be given an editorial or opinion article as well as {k} sentences from that article. 

For each sentence, give a keyword label to explain the author's argumentation strategy in that sentence. 
This means identifying the persuasive function of the sentence in the context of the broader argumentative structure and, if appropriate, identifying the kind of evidence being used.
Avoid summarizing the content or repeating the surface meaning. Your goal is to interpret each unit's argumentative role.
Think of how the sentence tries to influence the reader's belief, agreement, or perception, focus on describing what each unit is doing rhetorically.

Return a list of {k} labels in the format: 
[{{
  'sentence_idx': SENTENCE_IDX,
  'label': LABEL,
  'description': DESCRIPTION
}},
{{
    'sentence_idx': SENTENCE_IDX,
    'label': LABEL,
    'description': DESCRIPTION
}},
...
]

<article>
{article}
</article>

<sentences>
{sentences}
</sentences>

Your response:
"""

NEWS_CLUSTER_LABELING_PROMPT = '''You are a helpful assistant. I will give you a list of news headlines and summaries. You will summarize them and 
return a single, specific topic label and a description in the following forward: "Label": Description. 
Please condense them into a single, specific label. Be precise and concise. Ignore labels that are too generic.
Please return just one 2-3 word label and one description.

Here are some examples of how I want my outputs:
<examples>
output:
"Space Industry": These articles cover missions planned either by government or private companies into space.

output:
"Heart Health": The step covers medical advances impacting cardiovascular systems.
</examples>

Now it's your turn. Here are the article headlines and summaries:
<articles>
{articles_and_summaries}
</articles>

output:
'''

class EditorialLabelingResponse(BaseModel):
    """Response format for single sentence editorial labeling."""
    label: str
    description: str

class MultiSentenceLabelingItem(BaseModel):
    """Individual sentence labeling in multi-sentence response."""
    sentence_idx: int
    label: str
    description: str

class MultiSentenceLabelingResponse(BaseModel):
    """Response format for multi-sentence editorial labeling."""
    sentences: List[MultiSentenceLabelingItem]


class SimilarityPrompt(BaseModel):
    """Prompt for similarity task."""
    prompt: str
    label_1: str
    label_2: str
    description_1: str
    description_2: str

class SimilarityResponse(BaseModel):
    """Response format for similarity task."""
    pair_idx: int
    label: Literal['Yes', 'No']

class MultiSimilarityResponse(BaseModel):
    """Response format for multi-sentence similarity labeling."""
    pairs: List[SimilarityResponse]

# Tree Generation Prompts
# These prompts are used in create_trees.py for generating hierarchical tree structures
# from clusters of labeled data. They help summarize and label nodes in the tree.

KEYWORD_DEFINITION_NODE_PROMPT = """
You are a helpful assistant. I will give you a list of labels I wrote. You will summarize them and 
return a single, specific label and a description in the following forward: "Label": Description. 
Please condense them into a single, specific label. Be precise and concise. Ignore labels that are too generic.
Please return just one 2-3 word label and one description.

Here are some examples of how I want my outputs:
<examples>
output:
"Initial Condition": This step sets the starting point for a problem or calculation, establishing the initial state or value of a variable.

output:
"Reorganization": The step involves reorganizing or transforming data structures, such as lists or decks, to accurately reflect changes or updates.
</examples>

Now it's your turn. Here are the labels:
<labels>
{labels}
</labels>

output:
""" 

# Please focus specifically on methods of persuasion represented by the labels -- be specific!
KEYWORD_DEFINITION_NODE_PROMPT_WITH_EXAMPLES = """
You are a helpful assistant. I will give you a list of labels for classes of data points that I wrote, along with their definitions. 
Your task is to summarize the labels into a single 1-2 word label and description, capturing the meaning of these labels.
I will also give you examples of data points in each of classes.


Here are the labels:
<labels>
{labels}
</labels>

Here are the data points:
<data points>
{examples}
</data points>

Please summarize the labels into a single 1-2 word label and a description. 
Do not return a label that is too generic. Be precise, clear, and concise. 
Use the examples to help you better understand the labels and avoid returning a label that is too generic.

Your response:
""" 


SINGLE_SUMMARY_NODE_PROMPT = """
You are a helpful assistant. I will give you a label and a definition that I wrote.

Here is the label and the definition:

<label and definition>
{label_and_definition}
</label and definition>

Please summarize it in a single, specific label that captures the meaning of the definition. 
Be precise, clear, and concise. Do not be too generic. Return just the label.

Your response:
"""

class TreeNodeLabelResponse(BaseModel):
    """Response format for single label."""
    label: str
    description: str


ARE_SUMMARIES_IN_TEXT_PROMPT = """I will show you {k} pairs of summaries and text. 
The goal of the summaries are to capture this aspect of the text: {goal}. 
For each pair, please respond with 1 or 0 to indicate whether the summary captures the meaning of the text.
Respond with 0 if the summary is too broad or too vague to NOT be satisfied by the text.
Format your responses as a list the length of the number of summaries, where the ith entry is 1 if the ith summary is satisfied and 0 otherwise.

Here are the summary pairs:
<summary_pairs>
{summary_pairs}
</summary_pairs>

Your response:
"""

class AreSummariesInTextResponse(BaseModel):
    """Response format for are summaries in text task."""
    responses: List[Literal[1, 0]]
    
    # min_items: ClassVar[int] = 1
    # max_items: ClassVar[int] = 1
    # @field_validator('responses')
    # @classmethod
    # def validate_length(cls, v):
    #     if not (cls.min_items <= len(v) <= cls.max_items):
    #         raise ValueError(f'List must have between {cls.min_items} and {cls.max_items} items')
    #     return v


def make_summary_structure(min_len: int, max_len: int):
    return create_model(
        'AreSummariesInTextResponse',
        responses=(Annotated[list[int], Len(min_length=min_len, max_length=max_len)], ...)
    )


GENERATE_SUMMARIES_PROMPT = """
Here is a corpus of text samples, sorted from the lowest to the highest score.

<high_scoring_examples>
{high_scoring_examples}
</high_scoring_examples>

<low_scoring_examples>
{low_scoring_examples}
</low_scoring_examples>

Please propose {num_candidates} different short 2-4 word keyword summaries that DO summarize the text samples with high scores as they relate to {goal}, and do NOT summarize the text samples with lower scores.
The summaries should be concise and specific and focus on the meaning of the text samples as they relate to {goal}.
Each candidate summary should try to capture the meaning of ALL the text samples with high scores. I will select the best summary from your list, so try to make each summary as good as possible. I will discard VAGUE summaries
Your response is:
"""
# For each summary, return a label and a description of the label.
class GenerateSummariesResponse(BaseModel):
    """Response format for generate summaries task."""
    summaries: List[str]
    # summaries: List[TreeNodeLabelResponse]