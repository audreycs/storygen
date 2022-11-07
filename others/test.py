from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from numpy.linalg import norm

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

model = BertModel.from_pretrained('bert-base-uncased')

word1 = "storm"
word2 = "hurt"

inputs1 = tokenizer(word1, return_tensors="pt")
inputs2 = tokenizer(word2, return_tensors="pt")

outputs1 = model(**inputs1)
outputs2 = model(**inputs2)

A = outputs1.pooler_output.detach().numpy().squeeze()
B = outputs2.pooler_output.detach().numpy().squeeze()

cosine = np.dot(A,B)/(norm(A)*norm(B))

print(f"similarity: {cosine}")