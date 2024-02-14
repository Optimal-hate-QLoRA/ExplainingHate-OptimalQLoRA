import os
import pandas as pd
import transformers
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, set_seed, BitsAndBytesConfig,
    TrainingArguments, HfArgumentParser
)
from datasets import load_dataset
import torch
import bitsandbytes as bnb
from huggingface_hub import login
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm.auto import tqdm

from peft import LoraConfig, get_peft_model, PeftConfig, PeftModel, prepare_model_for_kbit_training

from utils import Multiclass
from utils import format_prompt

from sentence_transformers import SentenceTransformer

import torch.nn.functional as F

import torch.nn as nn

from eval_args import ScriptArguments

parser = HfArgumentParser(ScriptArguments)
args = parser.parse_args_into_dataclasses()[0]

def inference_function(args):
    login(token=args.hf_token)

    base_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    model = PeftModel.from_pretrained(base_model, "jbrophy123/llama2_7B_full",
                                  adapter_name="full_adapter")

    model.load_adapter("jbrophy123/llama2_7B_forums",
                   adapter_name='forums_adapter')
    model.load_adapter("jbrophy123/llama2_7B_microblog",
                    adapter_name="tw_gab_adapter")
    model.load_adapter("jbrophy123/llama2_7B_wiki",
                    adapter_name='wiki_adapter')

    encoder = SentenceTransformer('all-MiniLM-L6-v2')

    adapter_mapping={0:'forums_adapter',
                     1:'wiki_adapter',
                     2:'tw_gab_adapter'}

    generation_config = model.generation_config
    generation_config.max_new_tokens = args.max_new_tokens
    generation_config.temperature = args.temperature
    generation_config.top_p = args.top_p
    generation_config.num_return_sequences = args.num_return_sequences
    generation_config.pad_token_id = tokenizer.eos_token_id
    generation_config.eos_token_id = tokenizer.eos_token_id
    generation_config.do_sample = True

    eval_str = args.eval_str

    str_emb = encoder.encode(eval_str)

    router = Multiclass()

    router.load_state_dict(torch.load('./model/sentence_embedding_classifier.pth'))

    eval_prompt=format_prompt(eval_str)

    router.eval()
    # Make predictions and get probabilities
    with torch.no_grad():
        logits = router(str_emb)
        probabilities = F.softmax(logits, dim=0)

    if args.top_2_routing:
        top_2_list=np.argsort(probabilities.numpy())[-2:]
        adapters=[adapter_mapping.get(i) for i in top_2_list]

        model.add_weighted_adapter(adapters=adapters,
                                                weights=[0.5, 0.5],
                                                adapter_name="weighted_adapter_k2",
                                                combination_type="linear"
                                                )
        model.set_adapter('weighted_adapter_k2')

    elif args.top_3_routing:
        model.add_weighted_adapter(adapters=list(adapter_mapping.values()),
                                                weights=[0.33, 0.33, 0.33],
                                                adapter_name="weighted_adapter_k3",
                                                combination_type="linear"
                                                )
        model.set_adapter('weighted_adapter_k3')
    
    elif args.top_1_routing:
        if args.top_1_routing_adapter is not None:
            if args.top_1_routing_adapter == 0:
                model.set_adapter('forums_adapter')
            elif args.top_1_routing_adapter == 1:
                model.set_adapter('wiki_adapter')
            else:
                model.set_adapter('tw_gab_adapter')
        else:
            raise ValueError("You must specify which adapter to use with --top_1_routing_adapter")
    
    elif args.full_adapter:
        model.set_weighted_adapter("full_adapter")
    else:
        model.add_weighted_adapter(adapters=list(adapter_mapping.values()),
                                    weights=probabilities,
                                    adapter_name='weighted_adapter_blending',
                                    combination_type='linear'
                                    )
        model.set_adapter('weighted_adapter_blending')

    device='cuda:0'

    encoding=tokenizer(eval_prompt, return_tensors='pt').to(device)
    with torch.inference_mode():
    outputs=model.generate(
        input_ids=encoding.input_ids,
        attention_mask=encoding.attention_mask,
        generation_config=generation_config
    )
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__=='__main__':
    inference_function(args)











        
