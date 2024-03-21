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

    # Adjust year based on the month, considering the year transition
    return [date(2023 if int(month) <= 5 else 2022, int(month), 1) for month in sorted_dates]


class TPPMIModel:
    """Time-Pointwise Mutual Information (TPPMI) model."""

    def __init__(self, ppmi_models: dict, dates="months"):
        """Initialize the TPPMIModel.

        Args:
            ppmi_models (dict): Dictionary of PPMIModel instances with dates as keys.
            target_words (list): List of target words for TPPMI representation.
        """
        self.ppmi_models = ppmi_models
        if dates == "months":
            self.dates = _convert_dates(list(ppmi_models.keys()))
            self.is_in_months = True
        else:
            self.dates = list(ppmi_models.keys())
            self.is_in_months = False
        # will throw error if ppmi-models do not have the same context-words
        self.context_words = self._validate_alignment()
        self.vocab = sorted(list(set().union(*[model.get_vocabulary() for model in ppmi_models.values()])))
        self._vocab_word2ind = self._create_word_index(self.vocab)
        self._context_word2ind = self._create_word_index(self.context_words)

    def get_tppmi(self, target_words: list, selected_months=None, smooth=False) -> dict:
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
            if self.is_in_months:
                selected_months = [date.month for date in self.dates]
            else:
                selected_months = self.dates
        target_words = self._validate_selected_words(target_words)
        # Create an empty dictionary to store the TPPMI matrices
        tppmi_dict = dict(zip(target_words, [None] * len(target_words)))

        for word in target_words:
            # create T x dim(embedding) matrix (shape of tppmi-matrix)

            if self.is_in_months:
                index = [f"{word}_{date.month:02d}" for date in self.dates
                         if date.month in selected_months]
            else:
                index = [f"{word}_{date}" for date in self.dates
                         if date in selected_months]

            word_vectors = pd.DataFrame(index=index,
                                        columns=self.context_words,
                                        dtype=float).sort_index(axis=1)
            # for each timestep
            for date in self.dates:

                if self.is_in_months:
                    condition = date.month in selected_months
                    key = f"{date.month:02d}"
                else:
                    condition = date in selected_months
                    key = date

                if condition:
                    # see which words from the vocab are present in this timestep
                    current_vocab = self.ppmi_models[key].get_vocabulary()
                    if word in current_vocab:
                        # Extract ppmi-vector for work k of timestep i
                        ppmi_vector = self.ppmi_models[key].get_word_vector(word)
                        word_vectors.loc[f"{word}_{key}"] = ppmi_vector
                    else:
                        print(f"{word} - not in vocab of timestep: " + key)
            # Fill missing values (NaN) with zeros in the TPPMI matrix
            tppmi_dict[word] = word_vectors.fillna(0)

        if smooth:
            self._smooth(target_words, tppmi_dict)

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

    def _smooth(self, target_words: list, tppmi_dict: dict):
        # Convert dates to a numeric scale (e.g., days since the first date)
        num_dates = np.array([(date - self.dates[0]).days for date in self.dates])

        for target_word in target_words:
            # Retrieve the TPPMI dataframe for the current target word
            tppmi_df = tppmi_dict[target_word]

            for context_word in self.context_words:
                # Extract the TPPMI values for the current context word across all dates
                y = tppmi_df[context_word].values

                # Indices of non-NaN values
                valid_indices = np.where(~np.isnan(y))[0]

                # Create a cubic spline only for non-NaN values
                spline = CubicSpline(num_dates[valid_indices], y[valid_indices])

                # Evaluate the spline across all dates to fill in NaN values with smoothed ones
                smoothed_values = spline(num_dates)

                # Update the TPPMI matrix with the smoothed values
                tppmi_dict[target_word][context_word] = smoothed_values

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


    def most_similar_words_by_vector(self, target_vector: np.ndarray, top_n=10) -> dict:
        """
       Finds and returns the most similar words to a given target vector across different PPMI models.

       Parameters:
       - target_vector (np.ndarray): The vector representation of the target word whose similar words are to be found.
       - top_n (int, optional): The number of top similar words to return for each PPMI model. Defaults to 10.

       Returns:
       dict: A dictionary where each key is a date corresponding to a specific PPMI model, and each value is a list
             of tuples containing the top_n most similar words to the target vector and their respective similarity scores.
       """

        similar_words = {}

        for date, model in self.ppmi_models.items():
            similar_words[date] = model.most_similar_words_by_vector(target_vector, top_n)

        return similar_words

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

    def get_context_words(self) -> list:
        """Get the context-words of the model.
           The context-words are the columns of the ppmi-models

        Returns:
            list: List of context words.
        """
        return self.context_words

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

    def is_in_vocab_of_timestep(self, word: str, timestep: str) -> bool:
        model = self.ppmi_models[timestep]
        return model.contains_in_vocab(word)

    def is_in_vocab(self, word: str) -> bool:
        """
        Checks if the given word is present in the intersection of the vocabularies of all PPMI models.

        This method determines if a word is common across all the Pointwise Mutual Information (PPMI) models
        associated with the TPPMIModel instance. It is useful for understanding if a word has been consistently
        captured across different time intervals in the data.

        Args:
            word (str): The word to check for presence in the vocabularies.

        Returns:
            bool: True if the word is present in the intersection of all model vocabularies, False otherwise.
        """
        # Extract vocabularies from all PPMI models
        all_vocabs = [set(model.get_vocabulary()) for model in self.ppmi_models.values()]

        # Find the intersection of all vocabularies
        common_vocab = set.intersection(*all_vocabs)

        # Check if the word is in the common vocabulary
        return word in common_vocab

