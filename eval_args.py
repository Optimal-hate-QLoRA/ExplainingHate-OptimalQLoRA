from dataclasses import dataclass, field
import os
from typing import Optional

@dataclass
class ScriptArguments:

    hf_token: str = field(
       metadata={"help": "Huggingface Token"}
    )

    max_new_tokens: Optional[int] = field(
        default = 200, metadata={"help":"Number of tokens to generate"}
    )

    temperature: Optional[float] = field(
        default = 0.7, metadata={"help":"Temperature for generation"}
    )

    top_p: Optional[float] = field(
        default = 0.9, metadata={'help':'If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.'}
    )

    num_return_sequences: Optional[int] = field(
        default = 1, metadata = {"help":"Number of responses per query"}
    )

    eval_str: Optional[str] = field(
        default = "Have a good day!", metadata = {"help":"The string you want the model to respond to"}
    )

    top_2_routing: Optional[bool] = field(
        default = False, metadata = {'help':'whether to use top k routing with k=2'}
    )

    top_3_routing: Optional[bool] = field(
        default = False, metadata = {'help':'whether to use top k routing with k=3'}
    )

    top_1_routing:  Optional[bool] = field(
        default = False, metadata = {'help':'whether to use just one adapter'}
    )

    top_1_routing_adapter: Optional[int] = field(
        default = 0, metadata = {'help':'which router to use if you choose k=1'}
    )

    full_adapter: Optional[bool] = field(
        default = False, metadata = {'help':'Whether to use the adapter trained on all the available data or not'}
    )

    