from data_loader import DataLoader
from prompt import *
import argparse
import numpy as np
import random
from local_kg import *
import torch
from datetime import datetime
import logging
import sys
from IPython.display import display
import ipywidgets as widgets

def set_logs():
    if not os.path.isdir('logs/'):
        os.mkdir('logs/')
    filename = datetime.now().strftime('log_%Y%m%d_%H_%M.log')

    logFormatter = logging.Formatter("%(levelname)s-%(asctime)s  %(message)s")
    rootLogger = logging.getLogger("requests")
    rootLogger.setLevel(logging.INFO)

    fileHandler = logging.FileHandler('logs/'+filename)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logging.Formatter(""))
    rootLogger.addHandler(consoleHandler)
    return rootLogger

def init_seed():
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

def show_args( args):
    for arg in vars(args):
        logger.info(f"{arg}={getattr(args, arg)}")
    logger.info("")

def run(logger, args):
    dataLoader = DataLoader()
    dataLoader.readfile(args.data_dir)
    kw_list = dataLoader.keyword_list[0]

    # kw_list = str(inp.value).strip().split(',')

    promt_sentence = promptGeneration(logger, args, kw_list)

    path, hubs, stem_to_words, nei_to_hub = build_kg(kw_list=kw_list)
    final_score = calculate_score(logger, args, path, hubs, stem_to_words, nei_to_hub)

    story = gpt3_generation(logger, args, promt_sentence, final_score, stem_to_words)

    logger.info("-----Story-----")
    logger.info(f"promt_sentence: {promt_sentence}")
    logger.info(story)

    gpt3_story = original_gpt3(logger, args, kw_list)

    logger.info("-----Original GPT-3 Story-----")
    logger.info(gpt3_story)


if __name__ == "__main__":
    logger = set_logs()

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="./data")
    parser.add_argument('--prompt_model', type=str, default="curie")
    parser.add_argument('--model', type=str, default="curie")
    parser.add_argument('--alpha', type=float, default=0.3)  # importance of other hub nodes
    parser.add_argument('--beta', type=float, default=0.8)   # importance of relevant words
    parser.add_argument('--freq_pen', type=float, default=0.6)
    parser.add_argument('--ps_pen', type=float, default=0.0)
    parser.add_argument('--temp', type=float, default=0)
    parser.add_argument('--top_p', type=float, default=1)
    args = parser.parse_args()

    show_args(args)
    init_seed()

    run(logger, args)