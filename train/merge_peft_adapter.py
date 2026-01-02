#!python
# -*- coding: utf-8 -*-
"""Merge a PEFT LoRA adapter into its base model."""

from dataclasses import dataclass, field
from typing import Optional

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, HfArgumentParser

@dataclass
class ScriptArguments:
    """
    Arguments for merging a LoRA adapter into the base model.
    """

    model_name: Optional[str] = field(default="./output/lora-alpaca", metadata={"help": "the model name"})
    output_name: Optional[str] = field(default=None, metadata={"help": "the model name"})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
print("script_args: ", script_args)

peft_model_id = script_args.model_name
peft_config = PeftConfig.from_pretrained(peft_model_id)
print("peft_config: ", peft_config)

# Load the base model weights that the adapter was trained on.
model = AutoModelForCausalLM.from_pretrained(
    peft_config.base_model_name_or_path,
    return_dict=True,
    torch_dtype=torch.float16,
)

# Attach adapter weights and merge them into the base model.
model = PeftModel.from_pretrained(model, peft_model_id)
model.eval()
model = model.merge_and_unload()

if script_args.output_name is None:
    output_name = f"{script_args.model_name}-adapter-merged"
    model.save_pretrained(output_name)
else:
    model.save_pretrained(f"{script_args.output_name}")
