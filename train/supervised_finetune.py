#!python
# -*- coding: utf-8 -*-
"""Supervised LoRA finetuning for Alpaca-style data."""

import os
import sys
from typing import List

# Ensure repo root is importable when running from train/.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import fire
import torch
import transformers

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
try:
    from peft import prepare_model_for_kbit_training
except ImportError:  # Fallback for older PEFT
    from peft import prepare_model_for_int8_training as prepare_model_for_kbit_training
from transformers import AutoTokenizer, LlamaForCausalLM

from data_loader import sft_dataloader


def train(
    base_model: str = "",
    data_path: str = "yahma/alpaca-cleaned",
    output_dir: str = "./output/lora-alpaca",
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    val_set_size: int = 2000,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    use_8bit: bool = True,
    allow_cpu_offload: bool = True,
    train_on_inputs: bool = True,
    add_eos_token: bool = False,
    group_by_length: bool = False,
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",
    wandb_log_model: str = "",
    resume_from_checkpoint: str = None,
    prompt_template_name: str = "alpaca",
):
    """Run supervised finetuning with LoRA adapters."""
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    # Configure device placement for single-GPU vs DDP.
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Configure optional Weights & Biases logging.
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    # Choose an 8-bit device map with optional CPU offload.
    num_gpus = torch.cuda.device_count()
    if use_8bit:
        if ddp:
            quant_device_map = device_map
        elif num_gpus > 1:
            quant_device_map = "auto"
        elif allow_cpu_offload:
            quant_device_map = {"model": 0, "lm_head": "cpu"}
        else:
            quant_device_map = None
    else:
        quant_device_map = device_map

    # Load the base model with quantization if requested.
    model_kwargs = {"torch_dtype": torch.float16}
    if quant_device_map is not None:
        model_kwargs["device_map"] = quant_device_map
    if allow_cpu_offload and use_8bit and not ddp and num_gpus > 1:
        max_gpu_mem = int(torch.cuda.get_device_properties(0).total_memory * 0.8)
        model_kwargs["max_memory"] = {0: max_gpu_mem, "cpu": "32GiB"}
    if use_8bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=use_8bit,
            llm_int8_enable_fp32_cpu_offload=allow_cpu_offload and use_8bit,
        )
        model_kwargs["quantization_config"] = bnb_config
    model = LlamaForCausalLM.from_pretrained(base_model, **model_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)

    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    if use_8bit:
        # Patch bitsandbytes state for PEFT LoRA injection.
        for module in model.modules():
            state = getattr(module, "state", None)
            if state is not None and not hasattr(state, "memory_efficient_backward"):
                state.memory_efficient_backward = False
        model = prepare_model_for_kbit_training(model)

    # Attach LoRA adapters to the base model.
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    if resume_from_checkpoint:
        # Resume from either full or adapter-only checkpoint.
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )
            resume_from_checkpoint = False
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
            print(f"model: ", type(model))
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()

    # Build tokenized train/eval datasets.
    data_loader = sft_dataloader.SFTDataLoader(
        data_path,
        cutoff_len,
        val_set_size,
        train_on_inputs,
        add_eos_token,
        prompt_template_name,
        tokenizer,
    )
    train_data, val_data = data_loader.load_data()

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    # Train with the HF Trainer API.
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if val_set_size > 0 else None,
            save_steps=200,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    # Save only adapter weights to keep checkpoints small.
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)

    print("\nIf there's a warning about missing keys above, please disregard :)")


if __name__ == "__main__":
    fire.Fire(train)
