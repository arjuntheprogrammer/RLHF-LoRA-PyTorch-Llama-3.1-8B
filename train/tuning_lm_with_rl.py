#!python
# -*- coding: utf-8 -*-

"""Run PPO tuning against a reward model using TRL.

This script demonstrates how to fine-tune a causal language model with
Proximal Policy Optimization (PPO) using the `trl` library. The workflow is:
- Build or load a dataset of prompts
- Generate model responses (rollouts) from the policy model
- Score responses using an external reward model (pipeline)
- Run PPO updates on the policy model (with optional LoRA adapters)
- Periodically save model checkpoints

Note: This script sets up a value head on the causal LM and uses a
PPOTrainer to manage rollouts, batching, and optimization steps.
"""
from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import Adafactor, AutoTokenizer, HfArgumentParser, pipeline, set_seed

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
try:
    from trl.core import LengthSampler
except ImportError:
    import random

    # Minimal fallback for older TRL builds.
    class LengthSampler:
        def __init__(self, min_value, max_value):
            self.min_value = min_value
            self.max_value = max_value

        def __call__(self):
            return random.randint(self.min_value, self.max_value)



@dataclass
class ScriptArguments:
    """
    CLI arguments for PPO tuning.
    """

    model_name: Optional[str] = field(default="", metadata={"help": "the model name"})
    tokenizer_name: Optional[str] = field(default="", metadata={"help": "the tokenizer name"})
    reward_model_name: Optional[str] = field(default="", metadata={"help": "the reward model name"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    output_max_length: Optional[int] = field(default=128, metadata={"help": "maximum length for generation"})
    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=32, metadata={"help": "the batch size"})
    ppo_epochs: Optional[int] = field(default=4, metadata={"help": "the number of ppo epochs"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    adafactor: Optional[bool] = field(default=False, metadata={"help": "whether to use the adafactor optimizer"})
    early_stopping: Optional[bool] = field(default=False, metadata={"help": "whether to early stop"})
    target_kl: Optional[float] = field(default=0.1, metadata={"help": "kl target for early stopping"})
    reward_baseline: Optional[float] = field(
        default=0.0,
        metadata={"help": "a baseline value that is subtracted from the reward"},
    )
    batched_gen: Optional[bool] = field(default=False, metadata={"help": "whether to use the batched text gen"})
    save_freq: Optional[int] = field(default=None, metadata={"help": "n steps to save the model"})
    output_dir: Optional[str] = field(
        default="./output/checkpoints/tuning_llama_rl/",
        metadata={"help": "n steps to save the model"},
    )
    seed: Optional[int] = field(default=0, metadata={"help": "the seed"})


parser = HfArgumentParser(ScriptArguments)
script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
print("script_args: ", script_args)

reward_model_name = script_args.reward_model_name

dataset_name = "./datasets/"
print("dataset_name: ", dataset_name)

# PPO hyperparameters and logging config.
config = PPOConfig(
    model_name=script_args.model_name,
    learning_rate=script_args.learning_rate,
    log_with=script_args.log_with,
    batch_size=script_args.batch_size,
    mini_batch_size=script_args.mini_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    optimize_cuda_cache=True,
    early_stopping=script_args.early_stopping,
    target_kl=script_args.target_kl,
    ppo_epochs=script_args.ppo_epochs,
    seed=script_args.seed,
)

train_dataset = load_dataset(dataset_name, split="train")
train_dataset = train_dataset.select(range(100))
sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 1, "truncation": True}

tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_name, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def build_dataset(tokenizer, dataset_name="lvwerra/stack-exchange-paired"):
    """Tokenize and filter the dataset into PPO-ready inputs.

    We convert each raw prompt into a text `query` and pre-tokenize it so the
    PPO trainer can efficiently generate responses from the policy. We filter
    out inputs that exceed the model's max context (512 tokens here).
    """
    ds = load_dataset(dataset_name, split="train")
    original_columns = ds.column_names
    num_proc = 24

    def preprocess_function(examples):
        new_examples = {
            "query": [],
            "input_ids": [],
        }
        for question in examples["user_input"]:
            # Build a clear prompt that the policy will continue from.
            query = "Question: " + question + "\n\nAnswer: "
            tokenized_question = tokenizer(query, truncation=True)
            new_examples["query"].append(query)
            new_examples["input_ids"].append(tokenized_question["input_ids"])

        return new_examples

    ds = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )
    # Filter by sequence length so generation stays within supported context.
    ds = ds.filter(lambda x: len(x["input_ids"]) < 512, batched=False)

    ds.set_format(type="torch")
    return ds


dataset = build_dataset(tokenizer, dataset_name=dataset_name)


def collator(data):
    """Convert a list of dicts into a batch of lists.

    PPOTrainer expects batches where each entry is a list of items (not tensors).
    This collator converts a list-of-dicts into a dict-of-lists for convenience.
    """
    return dict((key, [d[key] for d in data]) for key in data[0])


set_seed(config.seed)

# `Accelerator().local_process_index` gives the process-local device id which
# is useful when mapping the model to the correct device in distributed runs.
current_device = Accelerator().local_process_index

# Attach a new LoRA adapter for PPO updates. Using LoRA keeps the PPO
# optimization focused on a small set of adapter parameters, which is useful
# for faster and cheaper RL fine-tuning.
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    config.model_name,
    load_in_8bit=True,
    device_map={"": current_device},
    peft_config=lora_config,
)

print("finetune model: ", type(model))
print("finetune model is_loaded_in_8bit: ", model.is_loaded_in_8bit)

optimizer = None
if script_args.adafactor:
    optimizer = Adafactor(
        filter(lambda p: p.requires_grad, model.parameters()),
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        lr=config.learning_rate,
    )
ppo_trainer = PPOTrainer(
    config,
    model,
    ref_model=None,
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=collator,
    optimizer=optimizer,
)

device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"
print("device: ", device)

# Reward model pipeline used to score responses.
sentiment_pipe = pipeline(
    "sentiment-analysis",
    model=reward_model_name,
    device_map={"": current_device},
    model_kwargs={"load_in_8bit": True},
    tokenizer=tokenizer,
)

# Generation settings for PPO rollouts.
generation_kwargs = {
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": 100_000,
}
output_min_length = 32
output_max_length = script_args.output_max_length
output_length_sampler = LengthSampler(output_min_length, output_max_length)

for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    question_tensors = batch["input_ids"]

    # 1) Generate responses from the current policy (rollouts)
    response_tensors = ppo_trainer.generate(
        question_tensors,
        return_prompt=False,
        length_sampler=output_length_sampler,
        **generation_kwargs,
    )
    batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

    # 2) Score generated responses with the reward model pipeline
    texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
    # Convert pipeline outputs into reward tensors expected by PPO
    rewards = [torch.tensor(output[0]["score"] - script_args.reward_baseline) for output in pipe_outputs]

    # 3) Make a PPO optimization step using the questions, responses and rewards
    stats = ppo_trainer.step(question_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)

    # 4) Periodically save model checkpoints
    if script_args.save_freq and (epoch + 1) % script_args.save_freq == 0:
        ppo_trainer.save_pretrained(script_args.output_dir + f"step_{epoch}")
