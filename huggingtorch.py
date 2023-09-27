#!/usr/bin/env python3
import datetime
from typing import List, Dict, Tuple
import re
import torch

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

def select_vocab(tok_module, regex=u'[\u4e00-\u9fff]', take_first: int=200, pad_to: int=128) -> Tuple[Dict[int, int], Dict[int,int]]:
    new_vocab: Dict[int, str] = {}
    vocab = tok_module.get_vocab()
    sorted_vocab: List[Tuple[int, str]] = []
    for k,v in vocab.items():
        sorted_vocab.append((v,k))
    sorted_vocab.sort()

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

def carve_embedding_vector(mod_module, encode_map: Dict[int,int]) -> torch.nn.parameter.Parameter:
    """Carves out the necessary embeddings"""
    select_dims = torch.tensor(sorted(encode_map.keys()))
    embeddings = mod_module.get_input_embeddings().weight
    myembeddings = torch.index_select(embeddings, 0, select_dims)
    return torch.nn.Parameter(myembeddings)


def assign_new_embeddings(mod_module, new_embeddings) -> None:
    size = new_embeddings.shape[0]
    mod_module.resize_token_embeddings(size, 128) # 128 silences a warning, we ensure we are padded well.
    curr_emb = mod_module.get_input_embeddings()
    curr_emb.weight.data = new_embeddings.data
    mod_module.set_input_embeddings(curr_emb)


def enc_dec(mod_module, tok_module, input_str, encode_map: Dict[int, int] | None = None, decode_map: Dict[int,int] | None = None) -> None:
    inputs = tok_module.encode(input_str, return_tensors="pt").to("cpu")
    # Remap inputs if necessary
    if encode_map is not None:
        for i in range(len(inputs[0])):
            inputs[0][i] = encode_map[int(inputs[0][i])]
    
    # Encode
    first_time = datetime.datetime.now()
    outputs = mod_module.generate(inputs)
    later_time = datetime.datetime.now()
    print(mod_module.get_input_embeddings().weight.shape, later_time - first_time)
    # Decode if necessary
    if decode_map is not None:
        for i in range(len(outputs[0])):
            outputs[0][i] = decode_map[int(outputs[0][i])]
    # Print result
    print(tok_module.decode(outputs[0]))


def testme(inputxt: str, unicoderange: str=u'[\u4e00-\u9fff]') -> None:
    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m")
    model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m")
    # pure
    enc_dec(model, tokenizer, inputxt)

    ### CARVING ###
    encode_map, decode_map = select_vocab(tok_module=tokenizer, regex=unicoderange)
    new_emb = carve_embedding_vector(model, encode_map)
    assign_new_embeddings(model, new_emb)
    ### CARVING ###
    # Shortlsited model test
    enc_dec(model, tokenizer, inputxt, encode_map, decode_map)


if __name__ == "__main__":
    # Chinese
    testme("生命的意义", u'[\u4e00-\u9fff]')

    # Cyrillic
    testme("Най високият връх в България е", u'[\u0400-\u04FF]')

    # English
    testme("The meaning of life is", u'[\u0000-\u007F]')

    # English - Latin1
    testme("The meaning of life is", u'[\u0000-\u00FF]')
