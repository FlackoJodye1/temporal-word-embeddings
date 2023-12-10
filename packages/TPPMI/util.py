import pickle
from glob import glob
from pathlib import Path

import pandas as pd
from typing import List
from collections import Counter

import nltk
import scipy.sparse as sp
from nltk.corpus import stopwords

from ppmi_model import PPMIModel # packages.TPPMI.
from tppmi_model import TPPMIModel # packages.TPPMI.

def contruct_tppmi_from_files(path):

    number_of_context_words = 500
    ppmi_path = Path(path)

    ppmi_data_files = sorted(glob(str(ppmi_path  / "*.npz")))
    words_files = sorted(glob(str(ppmi_path  / "*.pkl")))

    # Split context-words from timestamped-vocabularies
    context_words_file = [path for path in words_files if "context-words" in path]
    ppmi_vocab_files = [path for path in words_files if "context-words" not in path]

    # Get ppmi-matrices and vocab
    ppmi_matrices = {}

    for filenames in zip(ppmi_vocab_files, ppmi_data_files):
        ppmi_matrix = sp.load_npz(filenames[1])
        with open(filenames[0], "rb") as f:
            vocab = pickle.load(f)
        key = filenames[0].split("ppmi-")[2][0:2]
        ppmi_matrices[key] = {"ppmi_matrix" : ppmi_matrix, "vocab": vocab}

    # Get common context-words
    with open(context_words_file[0], "rb") as f:
        context_words = pickle.load(f)

    print(ppmi_matrices.keys())

    # Create ppmi_model objects

    ppmi_models = {key: PPMIModel.construct_from_data(ppmi_data["ppmi_matrix"], ppmi_data["vocab"], context_words) for
                   key, ppmi_data in ppmi_matrices.items()}

    return TPPMIModel(ppmi_models)

def get_most_common_words(corpus: str, top_n: int, remove_stopwords: bool = False):
    """
    Extract the most common words from a specified column in a pandas dataframe.

    :param corpus: a string containing the whole corpus.
    :param top_n: Number of top common words to return.
    :return: A list of the most common top_n words.
    """

    # Split the string into words and create a counter object
    words = corpus.split()

    if remove_stopwords:
        english_stopwords = set(stopwords.words('english'))
        words = [word for word in words if word.lower() not in english_stopwords]

    word_counts = Counter(words)

    # Get the most common words
    common_words = word_counts.most_common(top_n)

    return [word for word, count in common_words]