#!/usr/bin/env python3

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-7b1")
model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-7b1")


# llama

tokenizer = AutoTokenizer.from_pretrained("/mnt/startiger0/llama/llama7b-hf")
model = AutoModelForCausalLM.from_pretrained("/mnt/startiger0/llama/llama7b-hf")
