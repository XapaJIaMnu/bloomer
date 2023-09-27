#!/usr/bin/env python3
from huggingtorch import *


tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m")
model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m")
print(model.dtype)


#### Carve out code here ###
encode_map, decode_map = select_vocab(tokenizer)
new_embeddings = carve_embedding_vector(model, encode_map)


#inputs = tokenizer.encode("Translate to English: Je t’aime.", return_tensors="pt").to("cuda")
inputs = tokenizer.encode("Translate to English: Je t’aime.", return_tensors="pt").to("cpu")
outputs = model.generate(inputs)
vocab = tokenizer.get_vocab() # Getting vocabulary
print(tokenizer.decode(outputs[0]))

#
input2 = tokenizer.encode("help me understand the meaning of life", return_tensors="pt").to("cpu")
output2 = model.generate(input2)
print(tokenizer.decode(output2[0]))
input3 = tokenizer.encode("the meaning of life is", return_tensors="pt").to("cpu")
first_time = datetime.datetime.now()
output3 = model.generate(input3)
later_time = datetime.datetime.now()
print("250k vocab:", later_time - first_time)
print(tokenizer.decode(output3[0]))
model.resize_token_embeddings(80000, 128)

first_time = datetime.datetime.now()
output3 = model.generate(input3)
later_time = datetime.datetime.now()
print("80k vocab:", later_time - first_time)
print(tokenizer.decode(output3[0]))

model.resize_token_embeddings(32000, 128)
first_time = datetime.datetime.now()
output3 = model.generate(input3)
later_time = datetime.datetime.now()
print("32k vocab:", later_time - first_time)
print(tokenizer.decode(output3[0]))


vocab = tokenizer.get_vocab()
sorted_vocab = []
for k,v in vocab.items():
    sorted_vocab.append((v,k))
sorted_vocab.sort()

chinese_vocab = {}

for word, id in vocab.items():
    if re.search(u'[\u4e00-\u9fff]', tokenizer.convert_tokens_to_string([word])):
        chinese_vocab[word] = id


input_zh = tokenizer.encode("哇 怎麼發生的 看醫生也好 有治療", return_tensors="pt").to("cpu")
print(input_zh)

# Now try GPU
#model = model.half()
#print(model.dtype)
#model_cuda = model.to("cuda")
#inputs_cuda = tokenizer.encode("Translate to English: Je t’aime.", return_tensors="pt").to("cuda")
#outputs = model_cuda.generate(inputs_cuda)
#print(tokenizer.decode(outputs[0]))
