import pandas as pd
from typing import List
from collections import Counter

import nltk
from nltk.corpus import stopwords


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