import requests
import spacy
from transformers import BertTokenizer, BertModel
import numpy as np
from numpy.linalg import norm
from gpt3_api import *
from transformers import logging
logging.set_verbosity_error()

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

discarded_rel = ["synonym", "similarto", "relatedto", "isa", "antonym", "externalurl", ]
expand_rel = {"motivatedbygoal": "motivated by goal",
              "hasprerequisite": "has prerequisite",
              "hasproperty": "has property",
              "causesdesire": "causes desire",
              "usedfor": "used for",
              "relatedto": "related to",
              "hassubevent": "has subevent",
              "distinctfrom": "distinct from",
              "atlocation": "at location",
              "isa": "is a",
              "antonym": "antonym",
              "formof": "form of",
              "capableof": "capable of",
              "hasa": "has a",
              "desires": "desires",
              "causes": "causes",
              "partof": "part of",
              "derivedfrom": "derived from",
              "createdby": "created by",
              "haslastsubevent": "has last subevent",
              "mannerof": "manner of",
              "madeof": "made of",
              "hascontext": "has context",
              "hasfirstsubevent": "has first subevent"
              }

def conceptNetTripleRetrival(concept):
    obj = requests.get('http://api.conceptnet.io/c/en/'+str(concept)).json()

    context = obj['edges']
    # logger.info(json.dumps(context[1], indent=4))

    triple_list = []

    count = 0
    for nei in context:
        triple = nei["@id"]
        start = nei["start"]["label"].lower()
        relation = nei["rel"]["label"].lower()
        end = nei["end"]["label"].lower()
        weight = float(nei["weight"])
        surfaceText = nei["surfaceText"]
        
        if relation in discarded_rel:
            continue
        
        relation = expand_rel[relation]

        count += 1

        triple_list.append((start, relation, end, weight, surfaceText))

    return triple_list


def similariry(w1, w2):
    inputs1 = bert_tokenizer(w1, return_tensors="pt")
    inputs2 = bert_tokenizer(w2, return_tensors="pt")

    outputs1 = bert_model(**inputs1)
    outputs2 = bert_model(**inputs2)

    A = outputs1.pooler_output.detach().numpy().squeeze()
    B = outputs2.pooler_output.detach().numpy().squeeze()

    cosine = np.dot(A,B)/(norm(A)*norm(B))
    return cosine

def promptGeneration(logger, kwlist, prompt_model):
    first_kw = kwlist[0]

    nlp = spacy.load("en_core_web_lg")
    doc = nlp(first_kw)
    
    words = []
    for token in doc:
        if token.pos_ == "NOUN" or token.pos_ == "VERB" or token.pos_ == "ADJ":
            words.append(str((token)))

    related_words = dict()
    for w in words:
        triples = conceptNetTripleRetrival(w)
        for t in triples:
            if w in t[0]:
                r = t[2]
            else:
                r = t[0]
            
            sim = np.mean([similariry(r, w) for w in words])

            related_words[r] = sim

    final_rw = sorted(related_words.items(), key=lambda item: item[1], reverse=True)[0][0]

    prompt_words = words + [final_rw]
    logger.info(f"prompt words: {prompt_words}")
    prompt_sentence = gpt3_k2s(logger, prompt_words, prompt_model)

    return prompt_sentence