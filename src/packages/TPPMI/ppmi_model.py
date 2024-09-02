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

    def __init__(self, text_df, context_words, min_freq=0, ppmi_matrix=None, vocab=None, normalize=False):
        """Initialize the PPMIModel.

        Args:
            text_df (pd.DataFrame): DataFrame with 'text' column containing text data.
            min_freq (int): Minimum frequency for retaining words.
            ppmi_df (pd.DataFrame): Precomputed PPMI matrix DataFrame.
        """
        if text_df is not None:  # @classmethod construct_from_texts() enters here
            self._tokenized_corpus = self._process_and_tokenize(text_df, min_freq)
            self.context_words = context_words
            self.vocab = list(set(word for document in self._tokenized_corpus for word in document))
            self._vocab_word2ind = self._create_word_index(self.vocab)
            self._context_word2ind = self._create_word_index(context_words)

            # Create PPMI-Matrix
            row = np.zeros((1, len(self.context_words)))[0]  # dim(embedding) x 1
            self.ppmi_matrix = np.array([row for _ in range(len(self.vocab))])  # dim(embedding) x V
            self._ppmi_matrix_exists = False
        elif ppmi_matrix is not None:  # @classmethod construct_from_data() enters here
            self.vocab = vocab
            self.context_words = context_words
            self._vocab_word2ind = self._create_word_index(vocab)
            self._context_word2ind = self._create_word_index(context_words)

            self.ppmi_matrix = ppmi_matrix.toarray()
            if normalize:
                self._l2_normalize()
            self._ppmi_matrix_exists = True
        else:
            raise ValueError("Either a dataframe containing texts or a dataframe containing ppmi-matrices has to be "
                             "provided.")

    @classmethod
    def construct_from_data(cls, ppmi_matrix: sp.csr_matrix, vocab, context_words, min_freq=0, normalize=False):
        """Construct PPMIModel from precomputed PPMI matrix (DataFrame).

        Args:
            ppmi_df (pd.DataFrame): Precomputed PPMI matrix DataFrame.

        Returns:
            PPMIModel: Constructed PPMIModel instance.
            :param vocab: vocabulary of the PMIModel instance.
            :param ppmi_matrix:
        """
        return cls(None, context_words, min_freq, ppmi_matrix, vocab, normalize=normalize)

    @classmethod
    def construct_from_texts(cls, text_df: pd.DataFrame, context_words, min_freq=0, normalize=False):
        """Construct PPMIModel from text DataFrame.
        """
        return cls(text_df, context_words, min_freq, None, None, normalize=normalize)

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

    def _create_word_index(self, word_list: list) -> dict:
        """
        Create a dictionary to map each word in the given list to its index.

        Args:
            word_list (list): A list of words, either vocabulary or context words.

        Returns:
            dict: A dictionary mapping words to their respective indices.
        """
        # Create an array of indices corresponding to the number of words in the given list
        indices = np.arange(len(word_list))

        # Create a dictionary that maps each word in the list to its corresponding index
        word2ind = dict(zip(word_list, indices))
        return word2ind

    def _compute_co_occurrence_matrix(self, window_size=5) -> np.ndarray:
        """Compute the co-occurrence matrix.

        Args:
            window_size (int): Size of the context window.

        Returns:
            np.ndarray: Co-occurrence matrix.
        """
        co_matrix = np.zeros((self.get_vocabulary_size(), len(self.context_words)), dtype=np.int32)

        for tokenized_text in self._tokenized_corpus:
            for index, center_word in enumerate(tokenized_text):

                # Create Window
                center_index = self._vocab_word2ind.get(center_word, -1)
                if center_index == -1:
                    print("out of vocab")
                    continue

                upper = min(index + window_size + 1, len(tokenized_text))
                lower = max(index - window_size, 0)

                # Do the job
                context_words = tokenized_text[lower:index] + tokenized_text[index + 1:upper]

                # Update Co-occurrence Matrix
                context_indices = [self._context_word2ind.get(context_word, -1) for context_word in context_words]
                for context_index in context_indices:
                    if context_index != -1:
                        co_matrix[center_index, context_index] += 1

        return co_matrix

    def compute_ppmi_matrix(self, window_size=5, epsilon=1e-10, normalize = False) -> np.ndarray:
        """Compute the Pointwise Mutual Information (PPMI) matrix with improved handling of zero probabilities.

        Args:
            window_size (int): Size of the context window.
            epsilon (float): Small constant to avoid division by zero and zero probabilities in log.

        Returns:
            np.ndarray: PPMI matrix.
        """
        if self._ppmi_matrix_exists:
            return self.ppmi_matrix
        else:
            co_matrix = self._compute_co_occurrence_matrix(window_size) + epsilon

            # Calculate probabilities with smoothing
            center_counts = co_matrix.sum(axis=1)
            prob_cols_given_row = (co_matrix.T / center_counts).T

            context_counts = co_matrix.sum(axis=0)
            prob_of_cols = context_counts / context_counts.sum()

            # Calculate PMI with a safe log
            ratio = prob_cols_given_row / prob_of_cols
            pmi = np.log(ratio.clip(min=epsilon))
            self._ppmi_matrix_exists = True
            self.ppmi_matrix = np.maximum(pmi, 0)
            if normalize:
                self._l2_normalize()
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
        """
        Get the vector representation of a word in the embedding space as a pandas' series.

        Args:
            word (str): The word for which the vector representation is required.

        Returns:
            pd.Series: The vector representation of the word.

        Raises:
            ValueError: If the word is not in the vocabulary.
        """
        # Check if the word exists in the vocabulary
        if word in self.vocab:
            # Retrieve the index of the word in the vocabulary
            word_index = self._vocab_word2ind[word]

            # Fetch the corresponding word vector from the PPMI matrix
            word_vector = self.ppmi_matrix[word_index]

            # Convert the word vector into a pandas Series for easy handling,
            # using the context words as the index
            return word_vector #pd.Series(word_vector, index=self.context_words)
        else:
            # If the word is not found in the vocabulary, raise an error
            raise ValueError(f"'{word}' is not in the vocabulary.")

    def most_similar_words_by_vector(self, vector: np.ndarray, top_n=10) -> list:
        """Get the n most similar words to a given vector based on cosine similarity.

        Assumes input vectors in the model and the target vector are already L2-normalized.

        Args:
            vector (np.ndarray): The target vector, expected to be normalized.
            top_n (int): Number of similar words to retrieve.

        Returns:
            list: List of tuples containing (similar_word, cosine_similarity_score).
        """
        if not isinstance(vector, np.ndarray):
            raise ValueError("Input must be a numpy array.")

        # Ensure the target vector is normalized and reshaped for dot product operation
        '''
        vector = vector.flatten()
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)'''

        # Compute cosine similarities in a batch operation
        similarities = np.dot(self.ppmi_matrix, vector.T).flatten()

        # Get the indices of the top_n most similar words
        top_indices = np.argsort(-similarities)[:top_n]

        # Retrieve the corresponding words and their similarity scores
        most_similar = [(self.vocab[i], similarities[i]) for i in top_indices]

        return most_similar

    def _cosine_similarity_of_vectors(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate the cosine similarity between two vectors."""
        # Ensure vectors are flattened to 1D to avoid shape issues
        vec1 = vec1.flatten()
        vec2 = vec2.flatten()

        dot_product = np.dot(vec1, vec2)
        return dot_product.item()

    def cosine_similarity(self, word1: str, word2: str) -> float:
        """Calculate the cosine similarity between two words in the embedding space."""
        if word1 not in self.vocab or word2 not in self.vocab:
            return 0
        if word1 not in self.vocab:
            raise ValueError(f"'{word1}' is not in the vocabulary.")
        if word2 not in self.vocab:
            raise ValueError(f"'{word2}' is not in the vocabulary.")

        vector1 = self.get_word_vector(word1)
        vector2 = self.get_word_vector(word2)

        if vector1 is None or vector2 is None:
            return 0.0

        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)

        if norm1 == 0 or norm2 == 0:
            # Handle division by zero (sparse vectors)
            return 0

        dot_product = np.dot(vector1, vector2)

        similarity = dot_product / (norm1 * norm2)
        return similarity

    def get_context_words(self):
        return self.context_words

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
        """Check if a word is in the vocabulary.

        Args:
            word (str): The word to check for in the vocabulary.
        Returns:
            bool: True if the word is found in the vocabulary, False otherwise.
        """
        return word in self.vocab

    def get_as_df(self) -> pd.DataFrame:
        """Get the PPMI matrix as a DataFrame.

        Returns:
            pd.DataFrame: PPMI matrix as a DataFrame.
        """
        return pd.DataFrame(data=self.ppmi_matrix, columns=self.context_words, index=self.vocab)

    def _l2_norm(m):
        """Return an L2-normalized version of a matrix.

        Parameters
        ----------
        m : np.array
            The matrix to normalize.

        Returns
        -------

        Note:
        This was directly copied from the Cade-Package
        """
        dist = np.sqrt((m ** 2).sum(-1))[..., np.newaxis]
        return (m / dist).astype(np.REAL)

    def set_word_vector(self, word: str, new_vector: np.ndarray):
        """
        Set or update the vector representation of a word in the PPMI matrix.

        Args:
            word (str): The word for which the vector representation is to be set or updated.
            new_vector (np.ndarray): The new vector representation for the word.

        Raises:
            ValueError: If the word is not in the vocabulary or if the new vector length
                        does not match the number of context words.
        """
        # Ensure the word is in the vocabulary
        if word not in self.vocab:
            raise ValueError(f"'{word}' is not in the vocabulary.")

        # Ensure the new vector has the correct shape
        if new_vector.shape[0] != len(self.context_words):
            raise ValueError("The length of the new vector does not match the number of context words.")

        # Retrieve the index of the word in the vocabulary
        word_index = self._vocab_word2ind[word]

        # Update the PPMI matrix with the new vector
        self.ppmi_matrix[word_index, :] = new_vector

    def save(self, month: str, path: Path):
        # save vocab
        with open(path / f"ppmi-{month}-01.pkl", "wb") as f:
            pickle.dump(self.vocab, f)
        # save ppmi-matrix
        sparse_ppmi_matrix = sp.csr_matrix(self.ppmi_matrix)
        sp.save_npz(path / f"ppmi-{month}-01.npz", sparse_ppmi_matrix)
        # print confirmation-info
        print(f"PPMI data for {month} saved successfully.")
        print(f"Vocabulary Size: {self.get_vocabulary_size()}")

    def _l2_normalize(self):
        norms = np.linalg.norm(self.ppmi_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        self.ppmi_matrix = self.ppmi_matrix / norms
