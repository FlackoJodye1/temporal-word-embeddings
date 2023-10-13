from pathlib import Path

import nltk
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
from heapq import nlargest
from collections import Counter


def remove_infrequent_words(tokenized_corpus: list, min_freq: int) -> list:
    """Remove infrequent words from the corpus.
       Otherwise, calculating the matrices later might not be feasible (O(V^2))
    Args:
        tokenized_corpus (list of list): Tokenized text data.
        min_freq (int): Minimum frequency threshold for retaining words.

    Returns:
        list of list: Filtered corpus with infrequent words removed.
    """
    if min_freq <= 1:
        return tokenized_corpus
    word_frequencies = Counter(word for tokenized_text in tokenized_corpus for word in tokenized_text)
    filtered_corpus = [
        [word for word in tokenized_text if word_frequencies[word] >= min_freq]
        for tokenized_text in tokenized_corpus
    ]
    return filtered_corpus


class PPMIModel:
    """Pointwise Mutual Information (PPMI) model for text data."""

    def __init__(self, text_df, min_freq=0, ppmi_matrix=None, vocab=None):
        """Initialize the PPMIModel.

        Args:
            text_df (pd.DataFrame): DataFrame with 'text' column containing text data.
            min_freq (int): Minimum frequency for retaining words.
            ppmi_df (pd.DataFrame): Precomputed PPMI matrix DataFrame.
        """
        if text_df is not None:
            self._tokenized_corpus = self._process_and_tokenize(text_df, min_freq)
            self.vocab = sorted(list(set(word for document in self._tokenized_corpus for word in document)))
            self._word2ind = self._create_word2ind()

            # Create PPMI-Matrix
            row = np.zeros((1, len(self.vocab)))[0]  # V x 1
            self.ppmi_matrix = np.array([row for _ in range(len(self.vocab))])  # V x V
            self._ppmi_matrix_exists = False
        elif ppmi_matrix is not None:
            self.vocab = vocab
            self._word2ind = self._create_word2ind()
            self.ppmi_matrix = ppmi_matrix.toarray()
            self._ppmi_matrix_exists = True
        else:
            raise ValueError("Either a dataframe containing texts or a dataframe containing ppmi-matrices has to be "
                             "provided.")

    @classmethod
    def construct_from_data(cls, ppmi_matrix: sp.csr_matrix, vocab: list):
        """Construct PPMIModel from precomputed PPMI matrix (DataFrame).

        Args:
            ppmi_df (pd.DataFrame): Precomputed PPMI matrix DataFrame.

        Returns:
            PPMIModel: Constructed PPMIModel instance.
            :param vocab:
            :param ppmi_matrix:
        """
        return cls(None, 0, ppmi_matrix, vocab)

    @classmethod
    def construct_from_texts(cls, text_df: pd.DataFrame, min_freq=0):
        """Construct PPMIModel from text DataFrame.

        Args:
            text_df (pd.DataFrame): DataFrame with 'text' column containing text data.
            min_freq (int): Minimum frequency for retaining words.

        Returns:
            PPMIModel: Constructed PPMIModel instance.
        """
        return cls(text_df, min_freq, None)

    def _process_and_tokenize(self, text_df: pd.DataFrame, min_freq=0) -> list:
        tokenizer = nltk.tokenize.TreebankWordTokenizer()
        text_df.dropna(subset=["text"], inplace=True)
        tokenized_corpus = []

        for _, row in text_df.iterrows():
            # Only for testing purposes
            # tokenize and append
            tokenized_corpus.append(tokenizer.tokenize(row["text"]))
        tokenized_corpus = remove_infrequent_words(tokenized_corpus, min_freq)
        return tokenized_corpus

    def _create_word2ind(self) -> dict:
        indices = np.arange(len(self.vocab))
        word2ind = dict(zip(self.vocab, indices))

        return word2ind

    def _compute_co_occurrence_matrix(self, window_size=5) -> np.ndarray:
        """Compute the co-occurrence matrix.

        Args:
            window_size (int): Size of the context window.

        Returns:
            np.ndarray: Co-occurrence matrix.
        """
        row = np.zeros((1, self.get_vocabulary_size()))[0]  # V x 1
        co_matrix = np.array([row for _ in range(self.get_vocabulary_size())])  # V x V

        for tokenized_text in self._tokenized_corpus:
            for index, center_word in enumerate(tokenized_text):

                # Create Window
                center_index = self._word2ind[center_word]
                upper = index + window_size + 1 if index + window_size < len(tokenized_text) else (
                        len(tokenized_text) - 1)
                lower = index - window_size if index - window_size >= 0 else 0

                # Do the job
                context_words = tokenized_text[lower:upper]
                context_indices = [self._word2ind[context_word] for context_word in context_words]

                # Update Concurrence-Matrix
                for context_index in context_indices:
                    # we now filter out the center word itself, bc it was previously added to context_words
                    co_matrix[center_index, context_index] += 1 if center_index != context_index else 0

        return co_matrix

    def compute_ppmi_matrix(self, window_size=5) -> np.ndarray:
        """Compute the Pointwise Mutual Information (PPMI) matrix.

        Args:
            window_size (int): Size of the context window.

        Returns:
            np.ndarray: PPMI matrix.
        """
        if self._ppmi_matrix_exists:
            return self.ppmi_matrix
        else:
            co_matrix = self._compute_co_occurrence_matrix(window_size)

            # p(y|x) - calculate the probability of each context word given a center word
            center_counts = co_matrix.sum(axis=1).astype(float)
            prob_cols_given_row = (co_matrix.T / center_counts).T

            # p(y) - calculate the probability of each context word in the total set
            context_counts = co_matrix.sum(axis=0).astype(float)
            prob_of_cols = context_counts / sum(context_counts)

            # Calculate PMI: log( p(y|x) / p(y) )
            ratio = prob_cols_given_row / prob_of_cols
            ratio[ratio == 0] = 0.00001  # Avoid logarithm of zero

            pmi = np.log(ratio)
            self._ppmi_matrix_exists = True
            self.ppmi_matrix = np.maximum(pmi, 0)  # pmi -> ppmi
            return self.ppmi_matrix

    def most_similar_words(self, target_word: str, top_n=5) -> list:
        """Get the n most similar words to a given word based on cosine similarity.

        Args:
            word (str): The target word.
            n (int): Number of similar words to retrieve.

        Returns:
            list: List of tuples containing (similar_word, cosine_similarity_score).
            :param target_word: string
            :param top_n: number of words to return
        """
        if target_word not in self.vocab:
            raise ValueError(f"'{target_word}' is not in the vocabulary.")

        similar_words = []
        for vocab_word in self.vocab:
            if vocab_word != target_word:
                similarity = self.cosine_similarity(target_word, vocab_word)
                similar_words.append((vocab_word, similarity))

        similar_words = nlargest(top_n, similar_words, key=lambda x: x[1])
        return similar_words

    def get_word_vector(self, word: str) -> pd.Series:
        """Get the vector representation of a word in the embedding space as a pandas' series."""
        if word in self.vocab:
            word_index = self._word2ind[word]
            word_vector = self.ppmi_matrix[word_index]
            return pd.Series(word_vector, index=self.vocab)
        else:
            raise ValueError(f"'{word}' is not in the vocabulary.")

    def cosine_similarity(self, word1: str, word2: str) -> float:
        """Calculate the cosine similarity between two words in the embedding space."""

        if word1 not in self.vocab:
            return 0
            raise ValueError(f"'{word1}' is not in the vocabulary.")
        if word2 not in self.vocab:
            return 0
            raise ValueError(f"'{word2}' is not in the vocabulary.")

        vector1 = self.get_word_vector(word1)
        vector2 = self.get_word_vector(word2)

        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        dot_product = np.dot(vector1, vector2)

        if norm1 == 0 or norm2 == 0:
            return 0

        similarity = dot_product / (norm1 * norm2)
        return similarity

    def get_vocabulary(self):
        """Get the vocabulary of the model.

        Returns:
            list: List of vocabulary words.
        """
        return self.vocab

    def get_vocabulary_size(self) -> int:
        """Get the size of the vocabulary.

        Returns:
           int: Size of the vocabulary.
        """
        return len(self.vocab)

    def get_shape(self):
        """Get the shape of the PPMI matrix. (V x V)

        Returns:
           tuple: Shape of the PPMI matrix.
        """
        return self.ppmi_matrix.shape

    def contains_in_vocab(self, word: str) -> bool:
        return word in self.vocab

    def get_as_df(self) -> pd.DataFrame:
        """Get the PPMI matrix as a DataFrame.

        Returns:
            pd.DataFrame: PPMI matrix as a DataFrame.
        """
        return pd.DataFrame(data=self.ppmi_matrix, columns=self.vocab, index=self.vocab)

    def save(self, month: str, path: Path):
        with open(path / f"ppmi-{month}-01.pkl", "wb") as f:
            pickle.dump(self.vocab, f)
        sparse_ppmi_matrix = sp.csr_matrix(self.ppmi_matrix)
        sp.save_npz(path / f"ppmi-{month}-01.npz", sparse_ppmi_matrix)
        print(f"PPMI data for {month} saved successfully.")
        print(f"Vocabulary Size: {self.get_vocabulary_size()}")
