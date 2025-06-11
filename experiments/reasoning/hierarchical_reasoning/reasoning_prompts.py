"""
Prompt templates for hierarchical reasoning on chess problems.
"""

from typing import Annotated, List, Optional
from pydantic import BaseModel, create_model
from annotated_types import Len, Literal

CHOOSE_EXAMPLES_PROMPT = """You are a chess expert. 

I am trying to find good examples of thought patterns for a chess problem.

Here is the thought pattern I am looking for:

<thought_pattern>
{thought_pattern}
</thought_pattern>

Here are some examples of thought patterns:

<examples>
{examples}
</examples>

Choose the best 5 examples from the list above.
If the list contains less than 5 examples, choose all of them.
You can rewrite the examples you choose to make them more specific or clearer if you think that is helpful, 
but don't change the meaning of the examples or add any new information. 
Your response:
"""

FOLLOW_HIGH_LEVEL_START = """You are a chess expert. Here is a problem you've been trying to solve:

<problem>
{problem}
</problem>

THINK first about how to solve this problem. Try to follow this approach as you think:

<approach>
{approach}
</approach>

Here are some examples of how to follow this approach:

<approach_examples>
{approach_examples}
</approach_examples>

Do NOT follow the examples exactly. Instead, use them as a guide to think about how to solve the problem.

Output a SINGLE thought and do NOT output the answer. This thought should be a SINGLE step in your approach to solving the problem, like the examples. Do not include any other thoughts or information. STOP after the thought.

<thought>
"""

FOLLOW_HIGH_LEVEL_CONTINUATION = """You are a chess expert. Here is a problem you've been trying to solve:

<problem>
{problem}
</problem>

You are thinking about how to solve this problem. Here is what you've thought so far:

<thinking>
{thinking}
</thinking>

Now, think about what to do next. Try to follow this approach:

<approach>
{approach}
</approach>

Here are some examples of how to follow this approach:

<approach_examples>
{approach_examples}
</approach_examples>

Do NOT follow the examples exactly. Instead, use them as a guide to think about how to solve the problem.

Output a SINGLE thought. This thought should be a SINGLE step in your approach to solving the problem, like the examples. Do not include any other thoughts or information. STOP after the thought.
"""

FOLLOW_HIGH_LEVEL_FINAL = """You are a chess expert. Here is a problem you've been trying to solve:

<problem>
{problem}
</problem>

You have been thinking about how to solve this problem. Here are your thoughts so far:

<thinking>
{thinking}
</thinking>

Now, you're finished thinking. Output ONLY your answer and nothing else. 
Your answer should be in algebraic notation and enclosed in \\boxed{{}}. Please JUST output your answer, not any other text. Your answer now:
"""

CHOOSE_FIRST_NODE = """You are a chess expert. Here is a problem you've been trying to solve:

<problem>
{problem}
</problem>

You will start thinking about how to solve this problem. Choose the best approach from below to start thinking about your problem:

<options>
{options}
</options>

What approach will you take first? Choose the best option from above. Your response:
"""

CHOOSE_NEXT_NODE = """You are a chess expert. Here is a problem you've been trying to solve:

<problem>
{problem}
</problem>

You are thinking about how to solve this problem. Here is what you've thought so far:

<thinking>
{thinking}
</thinking>

You are thinking about what thoughts to have next. Here are some options:

<options>
{options}
</options>

What will you think next? Choose the best option from above. Your response:
"""

VANILLA_BASELINE = """You are a chess expert. Here is a chess problem you need to solve:

<problem>
{problem}
</problem>

Think step by step about how to solve this problem. Consider the current board state, your opponent's last move, and what the best response would be.

Your answer should be in algebraic notation and enclosed in \\boxed{{}}.
""" 


class NextThought(BaseModel):
    next_thought: str


class BestExamples(BaseModel):
    best_examples: Annotated[List[str], Len(min_length=1, max_length=5)]
    description: str


def make_labeling_structure(possible_labels: List[str]):
    """Create a Pydantic model for label choices"""
    return create_model(
        'NextThoughtType',
        next_thought_type=(Literal[tuple(possible_labels)], ...), # type: ignore
        description=(Optional[str], None),
    )
