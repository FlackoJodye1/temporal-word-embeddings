import numpy as np
import pandas as pd
from datetime import date
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.interpolate import CubicSpline


def _convert_dates(dates: list):
    # Define a custom order of months starting from June (6) to April (4)
    custom_month_order = [6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5]

    # Sort the list of months based on the custom order
    sorted_dates = sorted(dates, key=lambda month: custom_month_order.index(int(month)))

    # Create date objects with year 2022 for the sorted months
    return [date(2022, int(month), 1) for month in sorted_dates]


class TPPMIModel:
    """Time-Pointwise Mutual Information (TPPMI) model."""

    def __init__(self, ppmi_models: dict):
        """Initialize the TPPMIModel.

        Args:
            ppmi_models (dict): Dictionary of PPMIModel instances with dates as keys.
            target_words (list): List of target words for TPPMI representation.
        """
        self.ppmi_models = ppmi_models
        self.dates = _convert_dates(list(ppmi_models.keys()))
        # will throw error if ppmi-models do not have the same context-words
        self.context_words = self._validate_alignment()
        self.vocab = sorted(list(set().union(*[model.get_vocabulary() for model in ppmi_models.values()])))
        self._vocab_word2ind = self._create_word_index(self.vocab)
        self._context_word2ind = self._create_word_index(self.context_words)

    def get_tppmi(self, target_words: list, selected_months=None) -> dict:
        """
        Calculate Time-Pointwise Mutual Information (TPPMI) matrices for a list of target words over time intervals.

        Args:
            target_words (list): A list of target words for which TPPMI matrices will be computed.

        Returns:
            dict: A dictionary where keys are target words, and values are corresponding TPPMI matrices.

        This method calculates TPPMI matrices for a list of target words over different time intervals based on precomputed
        Pointwise Mutual Information (PPMI) models for each time interval. It handles cases where a word may not be present in
        the vocabulary for a specific time interval and fills missing values with zeros in the TPPMI matrices.
        :param selected_months: selected subset of months to filter for.
        """

        # Validate and filter the list of target words
        if selected_months is None:
            selected_months = [date.month for date in self.dates]
        target_words = self._validate_selected_words(target_words)
        # Create an empty dictionary to store the TPPMI matrices
        tppmi_dict = dict(zip(target_words, [None] * len(target_words)))

        for word in target_words:
            # create T x dim(embedding) matrix (shape of tppmi-matrix)
            word_vectors = pd.DataFrame(index=[f"{word}_{date.month:02d}" for date in self.dates
                                               if date.month in selected_months],
                                        columns=self.context_words,
                                        dtype=float).sort_index(axis=1)
            # for each timestep
            for date in self.dates:
                if date.month in selected_months:
                    # see which words from the vocab are present in this timestep
                    current_vocab = self.ppmi_models[f"{date.month:02d}"].get_vocabulary()
                    if word in current_vocab:
                        # Extract ppmi-vector for work k of timestep i
                        ppmi_vector = self.ppmi_models[f"{date.month:02d}"].get_word_vector(word)
                        word_vectors.loc[f"{word}_{date.month:02d}"] = ppmi_vector
                    else:
                        print(f"{word} - not in vocab of timestep: {date.month:02d}")
            # Fill missing values (NaN) with zeros in the TPPMI matrix
            tppmi_dict[word] = word_vectors.fillna(0)

        return tppmi_dict

    def get_2d_representation(self, target_words: list, selected_months=None, use_tsne=False) -> dict:
        """Get the 2D representation of TPPMI vectors.

        Args:
            use_tsne (bool): Use t-SNE for dimensionality reduction.

        Returns:
            dict: Dictionary with row names as keys and 2D vectors as values.
            :param selected_months: list of selected subset of months to filter for.
            :param use_tsne: true iff u want to use tsne, false for pca
            :param target_words: list of words to get the tppmi vectors for
        """

        # Take all months of the model if none are selected
        if selected_months is None:
            selected_months = [date.month for date in self.dates]

        tppmi_values = self.get_tppmi(target_words, selected_months=selected_months).values()

        concatenated_matrix = pd.concat(tppmi_values,
                                        axis=0, ignore_index=True)
        concatenated_matrix.columns = concatenated_matrix.columns.astype(str)

        if use_tsne:
            tsne = TSNE(n_components=2)
            vectors = tsne.fit_transform(concatenated_matrix)
        else:
            pca = PCA(n_components=2)
            vectors = pca.fit_transform(concatenated_matrix)
        row_names = [f"{word}_{date}" for word in target_words for date in selected_months]

        return {row_name: vector for row_name, vector in zip(row_names, vectors)}

    # currently not usable
    def _smooth(self, target_words: list, tppmi_dict: dict):

        numdates = len(self.dates)

        for target_word in target_words:
            for i, context_word in enumerate(self.vocab):
                y = tppmi_dict[target_word].loc[:, context_word]
                numdates_sub = self.dates

                # Fit a cubic spline to the data
                spline = CubicSpline(numdates_sub[~np.isnan(y)], y[~np.isnan(y)], bc_type='natural')

                # Evaluate the spline at all numdates_sub values
                spline_y = spline(numdates_sub)

                # Update the tppmi_list with the smoothed values
                tppmi_dict[target_word].loc[:, context_word] = spline_y

    def calculate_absolute_drift(self, word: str) -> float:
        """
        Calculate the absolute drift of a word over all time steps, accounting for different vector lengths.

        Args:
            word (str): The word for which to calculate the drift.

        Returns:
            float: The absolute drift value for the given word.
        """

        if word not in self.vocab:
            return 1000
            # raise ValueError(f"Word '{word}' is not in the vocabulary.")

        # Find the common vocabulary across all models
        common_vocab = set.intersection(*[set(model.get_vocabulary()) for model in self.ppmi_models.values()])

        if word not in common_vocab:
            raise ValueError(f"Word '{word}' is not present in the common vocabulary across time steps.")

        total_drift = 0.0

        word_vectors = pd.DataFrame(index=[f"{word}_{date.month:02d}" for date in self.dates
                                           ],
                                    columns=self.vocab,
                                    dtype=float).sort_index(axis=1)

        # Iterate through each time step
        for date in self.dates:
            ppmi_vector = self.ppmi_models[f"{date.month:02d}"].get_word_vector(word)
            word_vectors.loc[f"{word}_{date.month:02d}"] = ppmi_vector
            word_vectors.fillna(0, inplace=True)

        for i in range(len(self.dates)):
            if i + 1 < len(self.dates):
                date = self.dates[i]
                next_date = self.dates[i + 1]
                month_key = f"{word}_{date.month:02d}"
                next_month_key = f"{word}_{next_date.month:02d}"
                drift = np.linalg.norm(word_vectors.loc[next_month_key] - word_vectors.loc[month_key])
                total_drift += drift

        return round(total_drift, 2)

    def calculate_top_n_drift(self, top_n: int) -> dict:
        """
        Calculate the absolute drift for all words in the vocabulary and return the top_n words with the highest drift.

        Args:
            top_n (int): Number of top words to return based on their drift values.

        Returns:
            dict: A dictionary with the top_n words as keys and their corresponding absolute drift values as values, sorted in descending order of drift.
        """

        drift_values = {}

        # Iterate over each word in the vocabulary and calculate its drift
        for word in self.vocab:
            try:
                drift = self.calculate_absolute_drift(word)
                drift_values[word] = drift
            except ValueError as e:
                continue

        # Sort the drift values in descending order and select the top_n
        sorted_drift_values = dict(sorted(drift_values.items(), key=lambda item: item[1], reverse=True)[:top_n])

        return sorted_drift_values

    def most_similar_words(self, target_word: str, top_n=5) -> dict:
        """
        Find the most similar words to a target word over different time intervals based on precomputed PPMI models.

        Args:
            target_word (str): The target word for which similar words will be retrieved.
            top_n (int, optional): The number of top similar words to retrieve. Default is 5.

        Returns:
            dict: A dictionary where keys are composed of the target word and the month of the PPMI model, and values are
                  lists of similar words.

        This method finds the most similar words to a target word over different time intervals using precomputed
        Pointwise Mutual Information (PPMI) models. It returns a dictionary where keys indicate the target word and the
        corresponding month, and values are lists of similar words for each time interval.
        """

        if target_word not in self.vocab:
            raise ValueError(f"{target_word} is not in the vocabulary")

        similar_words = {}

        for date, model in self.ppmi_models.items():
            if model.contains_in_vocab(target_word):
                similar_words[f"{target_word}_{date}"] = model.most_similar_words(target_word, top_n)

        return similar_words

    def print_most_similar_words(self, target_word: str, top_n=5):
        """
       Print the most similar words to a target word over different time intervals based on precomputed PPMI models.

       Args:
           target_word (str): The target word for which similar words will be printed.
           top_n (int, optional): The number of top similar words to print. Default is 5.

       This method prints the most similar words to a target word over different time intervals using precomputed
       Pointwise Mutual Information (PPMI) models. It displays the target word and corresponding month, followed by the
       most similar words for each time interval.
       """
        words = self.most_similar_words(target_word, top_n=top_n)
        for key, values in words.items():
            print(f"Word: {key.split('_')[0]} - Month: {key.split('_')[1].capitalize()}")
            try:
                for value in values:
                    print(value)
            except KeyError:
                print(f"{target_word} not in vocab")
            print("--------------------------------")

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

    # ----------------------------------------------------------- #
    # ----------------------- Validators ------------------------ #
    # ----------------------------------------------------------- #

    def _validate_selected_words(self, target_words: list) -> list:

        if not isinstance(target_words, list):
            raise TypeError("The 'target_words' argument must be a list.")

        if len(set(target_words) & set(self.vocab)) < len(target_words):
            common_words = list(set(self.vocab) & set(target_words))
            if len(common_words) < 1:
                raise ValueError("None of the selected words are in the PPMI matrices!")
            else:
                print("Not all selected words are in PPMI matrices.")
                target_words = common_words
                print("Words changed to:", ", ".join(target_words))
                return target_words
        else:
            print("All words are contained in the vocabulary")
            return target_words

    def _validate_alignment(self) -> list:
        """
        Check if all ppmi_models in the dictionary have the same context words and return those.

        Raises:
            ValueError: If not all models have the same context words.

        Returns:
            list: A list of context words from the first model if all models have the same context words.
        """
        if not self.ppmi_models:
            raise ValueError("No PPMI models available for validation.")

        # Extract the first model's context words and sort them
        reference_context_words = sorted(next(iter(self.ppmi_models.values())).get_context_words())

        # Iterate over the models in the dictionary
        for model_key, ppmi_model in self.ppmi_models.items():
            current_context_words = sorted(ppmi_model.get_context_words())
            if current_context_words != reference_context_words:
                raise ValueError(
                    f"Context words mismatch found in model with key '{model_key}' compared to the first model.")

        return reference_context_words

    # ----------------------------------------------------------- #
    # ------------------------ Getters -------------------------- #
    # ----------------------------------------------------------- #

    def get_vocabulary(self) -> list:
        """Get the vocabulary of the model.

        Returns:
            list: List of vocabulary words.
        """
        return self.vocab

    def get_embedding_dim(self) -> int:
        """Get the embedding dimension of the model.

        Returns:
            int: Size of the embeddings.
        """
        return len(self.context_words)

    def get_vocabulary_size(self) -> int:
        """Get the size of the vocabulary.

        Returns:
            int: Size of the vocabulary.
        """
        return len(self.vocab)
