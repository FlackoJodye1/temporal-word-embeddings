import pickle
from glob import glob
from pathlib import Path
from random import sample
import scipy.sparse as sp

from collections import Counter
from nltk.corpus import stopwords

from ppmi_model import PPMIModel
from tppmi_model import TPPMIModel


def construct_tppmi_from_files(path):
    ppmi_path = Path(path)

    ppmi_data_files = sorted(glob(str(ppmi_path / "*.npz")))
    words_files = sorted(glob(str(ppmi_path / "*.pkl")))

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
        ppmi_matrices[key] = {"ppmi_matrix": ppmi_matrix, "vocab": vocab}

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
    :param remove_stopwords: true, if stopword-removal is to be performed.
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


def sample_from_most_common_words(corpus: str, top_n: int, sample_size: int, remove_stopwords: bool = False):
    """
    Sample a subset of the most common words from the corpus.

    :param corpus: a string containing the whole corpus.
    :param top_n: Number of top common words to consider.
    :param sample_size: Number of words to sample from the top common words.
    :param remove_stopwords: true, if stopword-removal is to be performed.
    :return: A list of sampled words.
    """

    # Get the top_n common words
    common_words = get_most_common_words(corpus, top_n, remove_stopwords)

    # Ensure sample_size is not greater than the length of common_words
    sample_size = min(sample_size, len(common_words))

    # Randomly sample words from the common words
    sampled_words = sample(common_words, sample_size)

    return sampled_words
