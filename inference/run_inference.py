#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate responses from base, SFT-adapter, or PPO-tuned models."""

import argparse
import os
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import AutoModelForCausalLMWithValueHead

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.prompter import Prompter


def build_quant_config(load_in_8bit: bool, allow_cpu_offload: bool):
    if not load_in_8bit:
        return None
    if not torch.cuda.is_available():
        raise RuntimeError("8-bit quantization requires CUDA.")
    return BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=allow_cpu_offload,
    )


def patch_bnb_state():
    try:
        from bitsandbytes.autograd._functions import MatmulLtState
    except ImportError:
        return
    if not hasattr(MatmulLtState, "memory_efficient_backward"):
        MatmulLtState.memory_efficient_backward = False


def resolve_dtype(load_in_8bit: bool):
    if load_in_8bit:
        return torch.float16
    return torch.float16 if torch.cuda.is_available() else torch.float32


def get_model_device(model):
    if hasattr(model, "pretrained_model"):
        return model.pretrained_model.device
    return next(model.parameters()).device


def load_base_model(model_name, device_map, quant_config, torch_dtype):
    kwargs = {"torch_dtype": torch_dtype}
    if device_map:
        kwargs["device_map"] = device_map
    if quant_config is not None:
        kwargs["quantization_config"] = quant_config
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    for module in model.modules():
        state = getattr(module, "state", None)
        if state is not None and not hasattr(state, "memory_efficient_backward"):
            state.memory_efficient_backward = False
    return model


def load_sft_model(base_model, adapter_path, merged_model_path, device_map, quant_config, torch_dtype):
    if merged_model_path:
        return load_base_model(merged_model_path, device_map, quant_config, torch_dtype)
    base = load_base_model(base_model, device_map, quant_config, torch_dtype)
    return PeftModel.from_pretrained(base, adapter_path)


def load_ppo_model(ppo_model_path, device_map, quant_config, torch_dtype):
    kwargs = {"torch_dtype": torch_dtype}
    if device_map:
        kwargs["device_map"] = device_map
    if quant_config is not None:
        kwargs["quantization_config"] = quant_config
    return AutoModelForCausalLMWithValueHead.from_pretrained(ppo_model_path, **kwargs)


def build_prompt(prompter, instruction, input_text):
    if input_text:
        return prompter.generate_prompt(instruction, input_text)
    return prompter.generate_prompt(instruction)


def main():
    parser = argparse.ArgumentParser(description="Run inference for base, SFT, or PPO models.")
    parser.add_argument("--mode", choices=["base", "sft", "ppo"], required=True)
    parser.add_argument("--model_name", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--adapter_path", default="./output/lora-alpaca")
    parser.add_argument("--merged_model_path", default="")
    parser.add_argument("--ppo_model_path", default="")
    parser.add_argument("--tokenizer_name", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--input", dest="input_text", default="")
    parser.add_argument("--template", default="alpaca")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--allow_cpu_offload", action="store_true")
    parser.add_argument("--show_prompt", action="store_true")
    args = parser.parse_args()

    if args.load_in_8bit:
        patch_bnb_state()

    quant_config = build_quant_config(args.load_in_8bit, args.allow_cpu_offload)
    torch_dtype = resolve_dtype(args.load_in_8bit)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.mode == "base":
        model = load_base_model(args.model_name, args.device_map, quant_config, torch_dtype)
    elif args.mode == "sft":
        model = load_sft_model(
            args.model_name,
            args.adapter_path,
            args.merged_model_path or None,
            args.device_map,
            quant_config,
            torch_dtype,
        )
    else:
        if not args.ppo_model_path:
            raise ValueError("Provide --ppo_model_path for PPO inference.")
        model = load_ppo_model(args.ppo_model_path, args.device_map, quant_config, torch_dtype)

    model.eval()

    prompter = Prompter(args.template)
    prompt = build_prompt(prompter, args.instruction, args.input_text)
    if args.show_prompt:
        print(prompt)

    inputs = tokenizer(prompt, return_tensors="pt")
    device = get_model_device(model)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    output_tokens = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        pad_token_id=tokenizer.pad_token_id,
    )
    decoded = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    if prompter.template["response_split"] in decoded:
        response = prompter.get_response(decoded)
    else:
        response = decoded

    print(response.strip())


if __name__ == "__main__":
    main()
