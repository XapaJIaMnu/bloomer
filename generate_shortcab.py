#!/usr/bin/env python3
import os
import sys
import re
import json
import warnings
import fire
import torch
import transformers
from tqdm import tqdm
from peft import PeftModel
from transformers import GenerationConfig
from transformers import AutoModelForCausalLM #, AutoModelForSeq2SeqLM
from transformers import AutoTokenizer, LlamaTokenizer
from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter
from typing import List, Dict, Tuple, Union
from collections import defaultdict

lora_base_map = {"bloom-560m":"bigscience/bloom-560m",
                 "bloom-1b7":"bigscience/bloom-1b7",
                 "bloom-7b1":"bigscience/bloom-7b1",
                 "llama-7b":"decapoda-research/llama-7b-hf"}

def carve_embedding_vector(mod_module, encode_map: Dict[int,int]) -> torch.nn.parameter.Parameter:
    """Carves out the necessary embeddings"""
    select_dims = torch.tensor(sorted(encode_map.keys()))
    embeddings = mod_module.get_input_embeddings().weight
    myembeddings = torch.index_select(embeddings, 0, select_dims)
    return torch.nn.Parameter(myembeddings)


def assign_new_embeddings(mod_module, new_embeddings) -> None:
    size = new_embeddings.shape[0]
    mod_module.resize_token_embeddings(size) # 128 silences a warning, we ensure we are padded well.
    curr_emb = mod_module.get_input_embeddings()
    curr_emb.weight.data = new_embeddings.data
    mod_module.set_input_embeddings(curr_emb)


def big_corpus_vocab(tok_module, mybook: str, regex= None, take_first: int=300, pad_to: int=128):
    new_vocab: Dict[int, int] = defaultdict(lambda: 1) # ViD: Frequency

    input_vector = tok_module.encode(mybook, return_tensors="pt").to("cpu")
    for num in input_vector[0]:
        num = int(num)
        new_vocab[num] = new_vocab[num] + 1

    # Padding code
    vocab = tok_module.get_vocab()
    sorted_vocab: List[Tuple[int, str]] = []
    for k,v in vocab.items():
        sorted_vocab.append((v,k))
    sorted_vocab.sort()
    print("Old vocab:", len(sorted_vocab))

    # Add additional unicode points here
    if regex is not None:
        for word, myid in vocab.items():
            if re.search(regex, tok_module.convert_tokens_to_string([word])):
                new_vocab[myid] = word
    # Now do the padding
    estimated_size = len(new_vocab) + take_first
    pad_value =  pad_to - estimated_size % pad_to

    # Takes care of the case where some of the selected tokens are part of our
    # padded size. We insert until we reach the desired number.
    curr_pad = 0
    i = 0
    while curr_pad < take_first + pad_value:
        vid, vstr = sorted_vocab[i]
        if vid not in new_vocab:
            new_vocab[vid] = vstr
            curr_pad = curr_pad + 1
        i = i + 1

    # For encoding: We need TrueVocabID -> Consecutive Num
    encode_map: Dict[int,int] = {}
    sorted_selected_vocab = sorted(list(new_vocab.items()))
    for i in range(len(sorted_selected_vocab)):
        realID, _ = sorted_selected_vocab[i]
        encode_map[realID] = i

    decode_map = {v: k for k, v in encode_map.items()}

    return (encode_map, decode_map)       

def main(
    load_8bit: bool = True,
    base_model: str = "",
    lora_weights: str = "",
    test_file: str = "",
    save_file: str = "",
    prompt_template: str = "alpaca",  # The prompt template to use, will default to alpaca.
):

    if lora_weights == "":
        assert base_model
        print("\n\n******WARNING: LoRA module is not specified. Loading only the base model for inference.******\n\n", flush=True)
    if lora_weights[-1] == "/":
        lora_weights = lora_weights[:-1]

    if not base_model:
        for suffix in lora_base_map:
            if lora_weights.endswith(suffix):
                base_model = lora_base_map[suffix]
                continue
        assert base_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        if torch.backends.mps.is_available():
            device = "mps"
    except:  # noqa: E722
        pass

    if test_file:
        test_lang = test_file.split(".jsonl")[0].split("_")[-1]
    if not save_file:
        save_file = "data/test-" + test_lang + "_decoded_by_" + lora_weights.split("/")[-1] + ".jsonl"
    if os.path.isfile(save_file):
        print("Test file's corresponding output exists, skipping now.", flush=True)
        print("Test: {}, Lora: {}".format(test_file, lora_weights))
        print("Save file: {}".format(save_file))
        return

    prompter = Prompter(prompt_template)

    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    except:
        if "llama" in base_model.lower():
            tokenizer = LlamaTokenizer.from_pretrained(base_model)
        else:
            raise NotImplementedError

    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto",
        )
        if lora_weights != "":
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                torch_dtype=torch.float16,
            )
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        if lora_weights != "":
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                device_map={"": device},
                torch_dtype=torch.float16,
            )
    else: # CPU
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        if lora_weights != "":
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                device_map={"": device},
            )
#    model.to(device)
    # unwind broken decapoda-research config
    #if "decapoda" in base_model:
    #    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    #    model.config.bos_token_id = 1
    #    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    if device == "cuda":
        print("Using " + str(torch.cuda.device_count())  + " GPU devices", flush=True)

    def read_data(filename):
        data = []
        with open(filename) as f:
            for line in f:
                line = json.loads(line.strip())
                data.append({"instruction": line["prompt"], "input": None})
        return data


    def evaluate(
        instruction,
        input=None,
        temperature=1,
        top_p=1,
        top_k=50,
        num_beams=4, # perhaps can experiment with this
        max_new_tokens=256,
        no_repeat_ngram_size=6,
        encode_map: Union[Dict[int, int], None] = None,
        decode_map: Union[Dict[int,int], None] = None,
        **kwargs,
    ):
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        if encode_map is not None:
            for i in range(len(inputs["input_ids"][0])):
                inputs["input_ids"][0][i] = encode_map[int(inputs["input_ids"][0][i])]
        input_ids = inputs["input_ids"].to(model.device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            no_repeat_ngram_size=no_repeat_ngram_size,
            **kwargs,
        )

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=False,
            )
        s = generation_output.sequences[0]
        if decode_map is not None:
            for i in range(len(s)):
                s[i] = decode_map[int(s[i])]
        output = tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return prompter.get_response(output)

    # testing prompts
    if test_file:
        test_lang = test_file.split(".jsonl")[0].split("_")[-1]
        assert len(test_lang) == 2
        data = read_data(test_file)
        txtin: str = ""
        for d in data:
            txtin = txtin + prompter.generate_prompt(d["instruction"], d["input"]) + "\n"

        if test_lang == "bg":
            unicode_range = u'[\u0400-\u04FF]'
        elif test_lang == "zh":
            unicode_range = u'[\u4e00-\u9fff]'
        elif test_lang == "en":
            unicode_range = u'[\u0000-\u007F]'
        else:
            raise("Unsupported language")
        encode_map, decode_map = big_corpus_vocab(tokenizer, txtin, regex = unicode_range)
        print("New vocab:", len(encode_map.keys()))
        new_emb = carve_embedding_vector(model, encode_map)
        assign_new_embeddings(model, new_emb)

        write_data = []
        for d in tqdm(data):
            instruction = d["instruction"]
            input = d["input"]
            response = evaluate(instruction, input, encode_map = encode_map, decode_map = decode_map).split("\n### ")[0].strip()
            d["response"] = response
            write_data.append(d)
        if not save_file:
            save_file = "data/test-" + test_lang + "_decoded_by_" + lora_weights.split("/")[-1] + ".jsonl"
        with open(save_file, "w") as out_f:
            for d in write_data:
                out_f.write(json.dumps(d, ensure_ascii=False, indent = 4) + "\n")
            print("Saved {}".format(save_file), flush=True)
    else:
        print("No test file provided, will test on a few pre-defined example questions.", flush=True)
        for instruction in [
            "What are the even numbers between 1 and 13?",
            "Please briefly introduce the animal alpaca.",
            "What is the meaning of life?",
        ]:
            print("Instruction:", instruction.strip())
            print("Response:", evaluate(instruction).split("\n### ")[0].strip())
            print()

        for instruction, input in zip(["Please write a sentence using the given word.",
                                       "Can you repeat the following phrase 3 times?"
                                      ],
                                      ["vicuna",
                                       "Scotland!"
                                      ]):
            print("Instruction:", instruction.strip())
            print("Input:", input.strip())
            print("Response:", evaluate(instruction, input).split("\n### ")[0].strip())
            print()


if __name__ == "__main__":
    try:
        fire.Fire(main)
    except Exception as e:
        print(e)
