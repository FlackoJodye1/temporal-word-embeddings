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
        self.vocab = sorted(list(set().union(*[model.get_vocabulary() for model in ppmi_models.values()])))
        self._word2ind = self._create_word2ind()

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
        # .month:02d
        row_names = [f"{word}_{date}" for word in target_words for date in selected_months]

        return {row_name: vector for row_name, vector in zip(row_names, vectors)}

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
        '''
        print("Selected Months")
        print(selected_months)
        print("Dates")
        print(self.dates)
        print("Months")
        for date in self.dates:
            print(date.month, end=' ')
        print("\n-------------")

        test_word = target_words[0]
        print("Test")
        for date in self.dates:
            if date.month in selected_months:
                print(f"{test_word}_{date.month:02d}")
        print("###############################")'''

        for word in target_words:
            # create T x V matrix (shape of tppmi-matrix)
            word_vectors = pd.DataFrame(index=[f"{word}_{date.month:02d}" for date in self.dates
                                               if date.month in selected_months],
                                        columns=self.vocab,
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

    # Create indexed dictionary = vocabulary
    def _create_word2ind(self) -> dict:
        """Create a vocabulary index dictionary.

        Returns:
            dict: Vocabulary word to index mapping.
        """
        indices = np.arange(len(self.vocab))
        word2ind = dict(zip(self.vocab, indices))
        return word2ind

    def get_vocabulary(self) -> list:
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
