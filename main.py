from data_loader import DataLoader
from prompt import *
import argparse
import numpy as np
import random

random.seed(42)
np.random.seed(42)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="./data")

    args = parser.parse_args()

    dataLoader = DataLoader()
    dataLoader.readfile(args.data_dir)

    promt_sentence = promptGeneration(dataLoader.keyword_list[0])

    