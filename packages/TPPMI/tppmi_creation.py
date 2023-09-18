import numpy as np
from collections import Counter

import nltk

nltk.download('punkt');
tokenizer = nltk.tokenize.TreebankWordTokenizer()


def compute_ppmi_matrix(co_occurrence_matrix):
    # p(y|x) - calculate the probability of each context word given a center word
    center_counts = co_occurrence_matrix.sum(axis=1).astype(float)
    prob_cols_given_row = (co_occurrence_matrix.T / center_counts).T

    # p(y) - calculate the probability of each context word in the total set
    context_counts = co_occurrence_matrix.sum(axis=0).astype(float)
    prob_of_cols = context_counts / sum(context_counts)

    # Calculate PMI: log( p(y|x) / p(y) )
    ratio = prob_cols_given_row / prob_of_cols
    ratio[ratio == 0] = 0.00001  # Avoid logarithm of zero
    pmi = np.log(ratio)
    ppmi = np.maximum(pmi, 0)  # pmi -> ppmi

    return ppmi


def compute_co_occurrence_matrix(tokenized_corpus, word2ind, window_size=4):
    vocab_size = len(word2ind)

    # Create Empty Co-Matrix
    row = np.zeros((1, vocab_size))[0]  # V x 1
    co_matrix = np.array([row for _ in range(vocab_size)])  # V x V

    for tokenized_text in tokenized_corpus:
        for index, center_word in enumerate(tokenized_text):

            # Create Window
            center_index = word2ind[center_word]
            upper = index + window_size + 1 if index + window_size < len(tokenized_text) else (len(tokenized_text) - 1)
            lower = index - window_size if index - window_size >= 0 else 0

            # Do the job
            context_words = tokenized_text[lower:upper]
            context_indices = [word2ind[context_word] for context_word in context_words]

            # Update Concurrence-Matrix
            for context_index in context_indices:
                # we now filter out the center word itself, bc it was previously added to context_words
                co_matrix[center_index, context_index] += 1 if center_index != context_index else 0

    return co_matrix


# Remove infrequent words from the corpus.
# Otherwise, calculating the matrices later is not feasible (O(V^2))
def remove_infrequent_words(tokenized_corpus, min_freq):
    # Calculate word frequencies
    word_frequencies = Counter(word for tokenized_text in tokenized_corpus for word in tokenized_text)

    # Create a filtered corpus by removing infrequent words
    filtered_corpus = [
        [word for word in tokenized_text if word_frequencies[word] >= min_freq]
        for tokenized_text in tokenized_corpus
    ]

    return filtered_corpus


def get_vocab(tokenized_corpus, min_freq=0):

    tokenized_corpus = remove_infrequent_words(tokenized_corpus, min_freq)

    vocab = sorted(list(set(word for document in tokenized_corpus for word in document)))

    return vocab


# Create indexed dictionary = vocabulary
def create_word2ind(vocab):
    indices = np.arange(len(vocab))
    word2ind = dict(zip(vocab, indices))

    return word2ind


def process_and_tokenize(df, cutoff=0):
    df.dropna(subset=["text"], inplace=True)
    tokenized_corpus = []

    counter = 0
    for _, row in df.iterrows():
        # Only for testing purposes
        if 0 < cutoff < counter:
            break
        # tokenize and append
        tokenized_corpus.append(tokenizer.tokenize(row["text"]))
        counter += 1
    return tokenized_corpus
