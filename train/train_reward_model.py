#!python
# -*- coding: utf-8 -*-
"""Train a reward model with pairwise preference data.

This script loads pairwise preference datasets where each example consists of
two responses (j, k) for the same prompt. The model is trained so that the
preferred response (j) receives a higher scalar reward than the dispreferred
response (k). We use a pairwise logistic loss:
    loss = -logsigmoid(score_j - score_k)

Supported features:
- LLaMA support with bitsandbytes quantization (8-bit) and optional CPU offload
- Generic Hugging Face sequence classification models as a fallback
- LoRA adapters for parameter-efficient fine-tuning
- Custom data collator that builds paired batches (input_j, input_k)
"""

import os
import sys

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

# Ensure repo root is importable when running from train/.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import evaluate
import numpy as np
import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
try:
    from peft import prepare_model_for_kbit_training
except ImportError:  # Fallback for older PEFT
    from peft import prepare_model_for_int8_training as prepare_model_for_kbit_training
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.utils import PaddingStrategy
from transformers import LlamaForSequenceClassification

from data_loader import rm_dataloader

@dataclass
class ScriptArguments:
    """
    CLI arguments for reward model training.
    """

    local_rank: Optional[int] = field(
        default=-1, metadata={"help": "Used for multi-gpu"})
    resume_from_checkpoint: Optional[bool] = field(
        default=False,
        metadata={"help": "If you want to resume training where it left off."},
    )
    deepspeed: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to deepspeed config if using deepspeed. You may need this if the model that you want to train doesn't fit on a single GPU."
        },
    )
    per_device_train_batch_size: Optional[int] = field(default=4)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=1)
    learning_rate: Optional[float] = field(default=2e-5)
    weight_decay: Optional[int] = field(default=0.001)
    model_name: Optional[str] = field(
        default="gpt2",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    bf16: Optional[bool] = field(
        default=True,
        metadata={
            "help": "This essentially cuts the training time in half if you want to sacrifice a little precision and have a supported GPU."
        },
    )
    num_train_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    train_subset: Optional[int] = field(
        default=100000,
        metadata={"help": "The size of the subset of the training data to use"},
    )
    eval_subset: Optional[int] = field(
        default=50000,
        metadata={"help": "The size of the subset of the eval data to use"},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables gradient checkpointing."},
    )
    optim: Optional[str] = field(
        default="adamw_hf",
        metadata={"help": "Enables gradient checkpointing."},
    )
    lr_scheduler_type: Optional[str] = field(
        default="linear",
        metadata={"help": "The lr scheduler"},
    )


# Parse command-line arguments into a `ScriptArguments` dataclass.
parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

# Configuration flags to control memory optimizations.
# `use_8bit` enables 8-bit model loading (via bitsandbytes) to reduce memory
# usage; `allow_cpu_offload` allows moving some parameters to CPU when helpful.
use_8bit = True
allow_cpu_offload = True


dataset_name = "./datasets/"
print("dataset_name: ", dataset_name)

model_name_split = script_args.model_name.split("/")[-1]
print("model_name_split: ", model_name_split)
output_name = os.path.join(
    "output",
    f"reward_model_{model_name_split}_{script_args.train_subset}_{script_args.learning_rate}",
)
# TrainingArguments must be set before model init when using DeepSpeed.
training_args = TrainingArguments(
    output_dir=output_name,
    learning_rate=script_args.learning_rate,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    num_train_epochs=script_args.num_train_epochs,
    weight_decay=script_args.weight_decay,
    bf16=script_args.bf16,
    evaluation_strategy="steps",
    eval_steps=200,
    save_strategy="no",
    save_steps=200,
    save_total_limit=2,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    gradient_checkpointing=script_args.gradient_checkpointing,
    deepspeed=script_args.deepspeed,
    remove_unused_columns=False,
    label_names=[],
    logging_strategy="steps",
    logging_steps=10,
    optim=script_args.optim,
    lr_scheduler_type=script_args.lr_scheduler_type,
)

# Initialize tokenizer. `trust_remote_code=True` allows model-specific
# tokenizer implementations (useful for LLaMA forks).
tokenizer = AutoTokenizer.from_pretrained(
    script_args.model_name, use_fast=True, trust_remote_code=True
)

# Some models do not define a pad token; default to EOS so padding works
# consistently during batch collation and sequence classification.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# Device mapping: default to automatic placement; adjust for DDP (Distributed
# Data Parallel) if WORLD_SIZE indicates multi-process training. When using DDP
# we pin the model to the local rank to avoid cross-device placement issues.
device_map = "auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
print("device_map: ", device_map)

# Model loading: we special-case LLaMA-based models to support
# bitsandbytes (8-bit) quantization and optional CPU offload. The model is
# loaded as a SequenceClassification head with a single scalar output (num_labels=1),
# which we interpret as the reward score for a response.
if "llama" in script_args.model_name:
    if use_8bit:
        from transformers import BitsAndBytesConfig
        # In some single-GPU cases, offload the scoring head to CPU to reduce GPU RAM.
        if allow_cpu_offload and not ddp and torch.cuda.device_count() == 1:
            device_map = {"model": 0, "score": "cpu"}
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=allow_cpu_offload,
        )
        # Load LLaMA reward model with 8-bit quantization.
        model = LlamaForSequenceClassification.from_pretrained(
            script_args.model_name,
            num_labels=1,
            torch_dtype=torch.float16,
            device_map=device_map,
            quantization_config=bnb_config,
        )
    else:
        # Standard FP16 load for LLaMA.
        model = LlamaForSequenceClassification.from_pretrained(
            script_args.model_name,
            num_labels=1,
            torch_dtype=torch.float16,
            device_map=device_map,
        )
else:
    # Fallback for non-LLaMA models; still load as a single-output sequence
    # classification model suitable for scalar reward prediction.
    model = AutoModelForSequenceClassification.from_pretrained(
        script_args.model_name,
        num_labels=1,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
        trust_remote_code=True,
    )

# Some model implementations expose a `state` object used by flash attention
# or other low-level kernels. Ensure it has `memory_efficient_backward` set so
# training works correctly when mixed precision/gradient checkpointing is used.
for module in model.modules():
    state = getattr(module, "state", None)
    if state is not None and not hasattr(state, "memory_efficient_backward"):
        state.memory_efficient_backward = False

# Prepare model for k-bit training (e.g., int8/8-bit). This rewires layers so they
# can be fine-tuned in low precision. We'll attach LoRA adapters next to do
# parameter-efficient tuning instead of updating all model weights.
model = prepare_model_for_kbit_training(model)

# Configure LoRA for parameter-efficient fine-tuning. LoRA injects
# low-rank adapters into attention/linear layers so that most base model
# weights remain frozen; only small adapter parameters are updated.
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,            # LoRA rank
    lora_alpha=16,  # scaling factor
    lora_dropout=0.05,
    bias="none",
)

# Apply LoRA adapters to the model which were prepared for k-bit training.
model = get_peft_model(model, peft_config)

# Print a summary of trainable parameters so we can confirm adapters were applied.
model.print_trainable_parameters()

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id
model.config.use_cache = not script_args.gradient_checkpointing
num_proc = 24

# Load the pairwise reward dataset. The loader returns tokenized examples
# where each example contains two alternatives: fields like `input_ids_j` and
# `input_ids_k` correspond to the pair (preferred, dispreferred).
reward_dataloader = rm_dataloader.RewardDataLoader(
    dataset_name,
    script_args.train_subset,
    script_args.eval_subset,
    num_proc,
    tokenizer,
)
train_dataset, eval_dataset = reward_dataloader.load_data()

@dataclass
class RewardDataCollatorWithPadding:
    """Batch pairwise samples into tensors suitable for the reward model.

    Each input example is a dict containing tokenized pairs: `*_j` (preferred)
    and `*_k` (dispreferred). The collator pads each side independently and
    returns a dictionary with `input_ids_j`, `input_ids_k`, etc., ready for
    the Trainer to feed into the model.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Separate the batch into two lists: one for the 'j' responses and one
        # for the 'k' responses, so we can pad them independently.
        features_j = []
        features_k = []
        for feature in features:
            features_j.append(
                {
                    "input_ids": feature["input_ids_j"],
                    "attention_mask": feature["attention_mask_j"],
                }
            )
            features_k.append(
                {
                    "input_ids": feature["input_ids_k"],
                    "attention_mask": feature["attention_mask_k"],
                }
            )
        # Pad each side separately to produce aligned input tensors.
        batch_j = self.tokenizer.pad(
            features_j,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch_k = self.tokenizer.pad(
            features_k,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids_j": batch_j["input_ids"],
            "attention_mask_j": batch_j["attention_mask"],
            "input_ids_k": batch_k["input_ids"],
            "attention_mask_k": batch_k["attention_mask"],
            "return_loss": True,
        }
        return batch


accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    """Compute pairwise accuracy for reward predictions.

    `eval_pred` is a tuple (predictions, labels); predictions contain model
    outputs used to decide whether the model ranks j above k. We convert the
    raw outputs into binary decisions (argmax) and compare to zeros (meaning
    that j should be preferred, encoded as 0 here to match the evaluation API).
    """
    predictions, _ = eval_pred
    # predictions may have a shape that requires argmax across axis 0 to get
    # a binary decision per example indicating whether j > k.
    predictions = np.argmax(predictions, axis=0)
    labels = np.zeros(predictions.shape)
    return accuracy.compute(predictions=predictions, references=labels)


class RewardTrainer(Trainer):
    """Trainer that computes a pairwise preference loss between two responses.

    For each example we have a pair (j, k). The model produces scalar rewards
    rewards_j and rewards_k. We use the logistic loss `-logsigmoid(r_j - r_k)`
    so that the model is penalized when it ranks the dispreferred response higher.
    """

    def compute_loss(self, model, inputs, return_outputs=False):
        # Forward pass for both items in the pair; we expect the model to return
        # a tuple where the first item is the logits/scores (shape: batch x 1).
        rewards_j = model(
            input_ids=inputs["input_ids_j"], attention_mask=inputs["attention_mask_j"])[0]
        rewards_k = model(
            input_ids=inputs["input_ids_k"], attention_mask=inputs["attention_mask_k"])[0]
        # Pairwise logistic loss: encourage rewards_j > rewards_k
        loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()
        if return_outputs:
            return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
        return loss


# Create the Trainer with our custom collator and metrics. The collator
# ensures that each batch contains aligned pairwise tensors for j and k.
trainer = RewardTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    data_collator=RewardDataCollatorWithPadding(
        tokenizer=tokenizer, max_length=512, pad_to_multiple_of=8),
)

# When using gradient checkpointing we must disable the model's cache.
model.config.use_cache = False

# Start training (optionally resume from checkpoint).
trainer.train(script_args.resume_from_checkpoint)

# Save final adapter+model state (LoRA adapters are saved along with model
# configuration so they can be loaded later for inference or further tuning).
print("Saving last checkpoint of the model")
model.save_pretrained(output_name, safe_serialization=False)
