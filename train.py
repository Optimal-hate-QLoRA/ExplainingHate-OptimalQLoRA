import os

import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    HfArgumentParser
)
from datasets import load_dataset
import torch

import bitsandbytes as bnb
from huggingface_hub import login, HfFolder

from trl import SFTTrainer

from utils import print_trainable_parameters, find_all_linear_names

from train_args import ScriptArguments

from peft import LoraConfig, get_peft_model, PeftConfig, PeftModel, prepare_model_for_kbit_training


parser = HfArgumentParser(ScriptArguments)
args = parser.parse_args_into_dataclasses()[0]

def training_function(args):
    #log in to Huggingface
    login(token=args.hf_token)

    # set seed
    set_seed(args.seed)

    #define the data path
    data_path=args.data_path

    dataset = load_dataset("json", data_files=data_path)

    # load model from the hub with a bnb config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    #Import model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        use_cache=False,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True
    )

    # save tokenizer for easy inference
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token=tokenizer.eos_token
    tokenizer.padding_side='right'
    #prepare model for 4 bit training 
    model=prepare_model_for_kbit_training(model)
    #find all the linear projection layers in the model for applying LoRA
    modules=find_all_linear_names(model)
    config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias='none',
        task_type='CAUSAL_LM',
        target_modules=modules
    )
    #create PEFT object
    model=get_peft_model(model, config)
    # Define training args
    output_dir = args.output_dir
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim=args.optim,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        bf16=False,
        max_grad_norm=args.max_grad_norm,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        group_by_length=True,
        lr_scheduler_type=args.lr_scheduler_type,
        tf32=False,
        report_to="none",
        push_to_hub=False,
        max_steps = args.max_steps
    )
    #define trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset['train'],
        dataset_text_field="text",
        max_seq_length=2048,
        tokenizer=tokenizer,
        args=training_arguments
    )
    #convert normalization layers to float 32 for more stable training 
    for name, module in trainer.model.named_modules():
        if "norm" in name:
            module = module.to(torch.float32)

    print('starting training')

    trainer.train()

    print('LoRA training complete')
    # Save trained adapter 
    lora_dir = args.lora_dir
    trainer.model.save_pretrained(lora_dir, safe_serialization=False)
    
    print("saved lora adapters")

    

if __name__=='__main__':
    training_function(args)

