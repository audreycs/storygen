import spacy
from prompt import conceptNetTripleRetrival, similariry
import os
from nltk.stem import PorterStemmer
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')

import networkx as nx
from networkx.algorithms import tournament
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def build_kg(kw_list):
    nlp = spacy.load("en_core_web_lg")
    ps = PorterStemmer()

    words = set()

    for kw in kw_list:
        doc = nlp(kw)
        for token in doc:
            if token.pos_ == "NOUN" or token.pos_ == "VERB":
                words.add(str((token)))
    
    words = list(words)

    triples = []
    stem_to_words = defaultdict(list)
    nei_to_hub = dict()

    for w in words:
        w_triples = [ (temp[0], temp[1], temp[2])for temp in conceptNetTripleRetrival(w)]

        for t in w_triples:
            if w in t[0]:
                doc = nlp(t[2])
                for token in doc:
                    if token.pos_ == "NOUN" or token.pos_ == "VERB":
                        stem = ps.stem(str(token))
                        stem_to_words[stem].append(str(token))
                        triples.append((w, t[1], stem))
                        nei_to_hub[stem] = w
            else:
                doc = nlp(t[0])
                for token in doc:
                    if token.pos_ == "NOUN" or token.pos_ == "VERB":
                        stem = ps.stem(str(token))
                        stem_to_words[stem].append(str(token))
                        triples.append((stem, t[1], w))
                        nei_to_hub[stem] = w

    if not os.path.isdir('local_kgs/'):
        os.mkdir('local_kgs/')
    
    path = 'local_kgs/'+'kg.txt'

    with open(path, 'w', encoding='utf-8') as f:
        # f.write(', '.join(kw_list)+'\n')
        for t in triples:
            f.write(str(t[0])+'\t'+str(t[1])+'\t'+str(t[2])+'\n')

    return path, words, stem_to_words, nei_to_hub

def calculate_score(logger, path, hubs, stem_to_words, nei_to_hub, alpha_):
    G = nx.Graph()
    all_nodes = set(hubs)
    
    with open(path, "r", encoding='utf-8') as f:
        for line in f.readlines():
            s = line.strip().split('\t')
            G.add_edge(s[0], s[2], weight=1)
            all_nodes.add(s[0])
            all_nodes.add(s[2])
    
    G.add_nodes_from(all_nodes)

    logger.info(f"Hub nodes num: {len(hubs)}, Total nodes num: {len(all_nodes)}")
    non_hubs = all_nodes - set(hubs)

    logger.info(f"hub words: {hubs}")
    # logger.info(f"related words: {non_hubs}")
    
    unreachable = 100
    distance_dict = defaultdict(list)

    logger.info("")

    for h in hubs:
        logger.info(f"calculating distance to hub \"{h}\"")
        for n in non_hubs:
            if nx.has_path(G, n, h):
                shortest_path = nx.shortest_path(G, n, h)
                dis = len(shortest_path) - 1
                sim = similariry(stem_to_words[n][0], h)
                distance_dict[n].append((h, dis, sim, sim*1/dis))
            else:
                dis = unreachable
                sim = similariry(stem_to_words[n][0], h)
                distance_dict[n].append((h, dis, sim, sim*1/dis))
    
    hubs = list(hubs)
    non_hubs = list(non_hubs)

    hub2id = defaultdict(int)
    nonHubNode2id = defaultdict(int)
    for idx, h in enumerate(list(hubs)):
        hub2id[h] = idx
    for idx, n in enumerate(list(non_hubs)):
        nonHubNode2id[n] = idx
    
    sim_matrix = np.empty((len(non_hubs), len(hubs)))
    for n in distance_dict.keys():
        for item in distance_dict[n]:
            h = item[0]
            sim = item[3]
            sim_matrix[nonHubNode2id[n]][hub2id[h]] = sim

    # pd.set_option("display.precision", 4)
    sim_matrix = pd.DataFrame(sim_matrix, columns=hubs, index = non_hubs)

    final_score = dict()
    for n in non_hubs:
        idx = nonHubNode2id[n]
        idx_h = hub2id[nei_to_hub[n]]
        score = (sum(sim_matrix.values[idx]) - sim_matrix.values[idx][idx_h]) * alpha_ + sim_matrix.values[idx][idx_h]
        final_score[n] = score

    keys = list(final_score.keys())
    values = [final_score[k] for k in keys]
    norm_values = [(float(i)-min(values))/(max(values)-min(values)) for i in values]

    final_score = dict(zip(keys + hubs, norm_values + [1.0]*len(hubs)))

    df_final_score = pd.DataFrame({'final_score': norm_values}, index=keys)

    sim_matrix = pd.concat([sim_matrix, df_final_score], axis=1)
    logger.info(sim_matrix.head(6))

    # drawing network and save into file
    color_map = []
    for node in non_hubs:
        color_map.append(final_score[node]*10+1)
    pos = nx.spring_layout(G)
    nx.draw_networkx(G, pos=pos, nodelist=non_hubs, node_color=color_map, cmap = plt.cm.Blues, node_size=12, font_size=4, font_color='#404040')
    nx.draw_networkx(G, pos=pos, nodelist=hubs, node_color='red', node_size=12, font_size=4, font_color='#404040')
    plt.savefig("local_kgs/kg.png", format="PNG", dpi=600)
    plt.clf()
    
    return final_score
