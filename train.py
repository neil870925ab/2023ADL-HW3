import torch
import numpy as np
from tqdm import tqdm
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
import json
from utils import get_prompt, get_bnb_config
import argparse
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import load_dataset
import logging
from accelerate import Accelerator
import os
from torch.utils.data import DataLoader
import math

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def add_prompt(examples):
    examples['instruction'] = get_prompt(examples['instruction'])
    return examples

def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["instruction"],
        padding=True,
        truncation=True,
        max_length=128,
    )
    labels = tokenizer(
        examples['output'], 
        padding=True, 
        truncation=True,
        max_length=128,
    )
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="",
        help="Path to the checkpoint of Taiwan-LLM-7B-v2.0-chat. If not set, this script will use "
        "the checkpoint from Huggingface (revision = 5073b2bbc1aa5519acdc865e99832857ef47f7c9)."
    )
    parser.add_argument(
        "--train_file", 
        type=str, 
        default=None, 
        help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the saved PEFT checkpoint."
    )
    parser.add_argument(
        "--training_epoch",
        type=int,
        default=2,
        help="number of training epoch."
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    args = parser.parse_args()
    accelerator_log_kwargs = {}
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    logging.basicConfig(level=logging.INFO)

    # Load model
    bnb_config = get_bnb_config()

    if args.base_model_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model_path,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config
        )
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    else:
        model_name = "yentinglin/Taiwan-LLM-7B-v2.0-chat"
        revision = "5073b2bbc1aa5519acdc865e99832857ef47f7c9"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
        )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    
    config = LoraConfig(
	   r=4, 
	   lora_alpha=16, 
	   lora_dropout=0.05, 
	   bias="none", 
	   task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, config)
    #print_trainable_parameters(model)
    
    #load dataset
    data_file = {}
    if args.train_file is not None:
        data_file = args.train_file
    extension = args.train_file.split(".")[-1]
    dataset = load_dataset(extension, data_files=data_file)
    dataset['train'] = dataset['train'].select(range(3000))

    dataset = dataset.map(lambda samples: add_prompt(samples))

    with accelerator.main_process_first():
        processed_datasets = dataset.map(
            preprocess_function, batched=True, remove_columns=dataset["train"].column_names
        )
    train_dataset = processed_datasets["train"]
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    device = accelerator.device
    model.to(device)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.training_epoch * num_update_steps_per_epoch
    
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=args.training_epoch * len(train_dataloader),
    )
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    
    #train
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logging.info("***** Running training *****")
    logging.info(f"  Num examples = {len(train_dataset)}")
    logging.info(f"  Num Epochs = {args.training_epoch}")
    logging.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logging.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logging.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)

    for epoch in range(args.training_epoch):
        model.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)


    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)

    

