import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# Code adapted from:
# ------------------------------------------------------------------------------
# Research Center for Computational Social Science (RC2S2)
# Author: Zsófia Rakovics
# (currently not publicly available)
# ------------------------------------------------------------------------------

def calc_tppmi(ppmi_list, dates=None, words=None, smooth=True):
    """
    Calculate tPPMI matrices for a given list of PPMI matrices.

    Args:
        ppmi_list (dict): A dictionary where keys are dates and values are PPMI matrices (DataFrames).
        dates (list, optional): List of dates to consider. Defaults to all dates in ppmi_list.
        words (list, optional): List of words to consider. Defaults to first two words from the first date's PPMI matrix.
        smooth (bool, optional): Whether to apply Laplace smoothing to PPMI values. Not used in this code.

    Returns:
        dict: A dictionary containing tPPMI matrices for selected words.

    """
    if not isinstance(ppmi_list, dict):
        raise ValueError("PPMI list could not be loaded!")

    # Set defaults if not provided
    if dates is None:
        dates = list(ppmi_list.keys())
        print("Dates set to names from ppmi_list.")
    if words is None:
        words = ppmi_list[dates[0]].iloc[:2, 0]
        print(f"Words set to {', '.join(words)}.")

    # create vocabulary (union of all time steps)
    # target_words = set().union(*[ppmi_list[date].iloc[:, 0] for date in ppmi_list])

    target_words = set()
    for i in range(len(dates)):
        target_words.update(ppmi_list[dates[i]].index.to_list())

    target_words = list(target_words)

    # Check if selected words are present in the PPMI matrices
    if len(set(words).intersection(target_words)) < len(words):
        if not set(words).intersection(target_words):
            raise ValueError("None of the selected words are in the PPMI matrices!")
        else:
            print("Not all selected words are in PPMI matrices.")
            words = list(set(words).intersection(target_words))
            print(f"Words changed to {', '.join(words)}.")

    tppmi_list = {}

    # Calculate tPPMI for each selected word
    for word in words:
        word_vectors = []
        for date in dates:
            ppmi_matrix = ppmi_list[date]
            if word in ppmi_matrix.iloc[:, 0].values:
                word_vectors.append(ppmi_matrix[ppmi_matrix.iloc[:, 0] == word].iloc[:, 1:].values)
            else:
                word_vectors.append(np.nan * np.ones(ppmi_matrix.shape[1] - 1))

        word_vectors = np.array(word_vectors)

        tppmi_list[word] = word_vectors

    return tppmi_list


# dimred_tppmi
def dimred_tppmi(tppmi_list, ndim=2, words=None, dates=None):
    if words is None:
        words = list(tppmi_list.keys())

    nrows = [tppmi_list[word].shape[0] for word in words]
    ncols = [tppmi_list[word].shape[1] for word in words]

    if len(set(ncols)) != 1:
        raise ValueError("TPPMI matrix dimensions are different!")

    tppmi_pca_list = {}

    for word in words:
        pca = PCA(n_components=ndim)
        tppmi_pca = pca.fit_transform(tppmi_list[word].reshape(tppmi_list[word].shape[0], -1))
        tppmi_pca_list[word] = tppmi_pca

    return tppmi_pca_list


# tppmi_pca_viz
def tppmi_pca_viz(tppmi_pca_list, date_x_offset=None, date_y_offset=None,
                  text_x_offset_start=None, text_y_offset_start=None,
                  text_x_offset_end=None, text_y_offset_end=None,
                  palcols=None, line_width=1, year=True, date_div=None,
                  text_div=None, date_start=None, date_end=None,
                  date_cex=None, text_cex=None):
    if not palcols:
        palcols = ["green", "purple", "blue"]

    all_dates = sorted(list(set().union(*[list(tppmi_pca_list[word].index) for word in tppmi_pca_list])))
    numcolors = len(all_dates)
    cols = plt.cm.rainbow(np.linspace(0, 1, numcolors))

    words = list(tppmi_pca_list.keys())

    for idx, word in enumerate(words):
        df = tppmi_pca_list[word]
        cols_sub = cols[[date in df.index for date in all_dates]]
        dates_sub = [date for date in all_dates if date in df.index]

        if idx == 0:
            plt.plot(df.iloc[:2, 0], df.iloc[:2, 1], col=cols_sub[:2], linewidth=line_width)
        else:
            plt.plot(df.iloc[:2, 0], df.iloc[:2, 1], col=cols_sub[:2], linewidth=line_width)
        for j in range(1, df.shape[0] - 1):
            plt.plot(df.iloc[j:j + 2, 0], df.iloc[j:j + 2, 1], col=cols_sub[j:j + 2], linewidth=line_width)

        if year:
            dates_sub_y = [date[:4] for date in dates_sub]
            for j, date in enumerate(dates_sub):
                if j % date_div == 0:
                    plt.text(df.iloc[j, 0] + date_x_offset, df.iloc[j, 1] + date_y_offset,
                             dates_sub_y[j], color=cols_sub[j], fontsize=date_cex)
        else:
            for j, date in enumerate(dates_sub):
                if j % date_div == 0:
                    plt.text(df.iloc[j, 0] + date_x_offset, df.iloc[j, 1] + date_y_offset,
                             date, color=cols_sub[j], fontsize=date_cex)

        plt.text(df.iloc[0, 0] + text_x_offset_start, df.iloc[0, 1] + text_y_offset_start,
                 word, color=cols_sub[0], fontsize=text_cex)
        for j in range(1, df.shape[0] - 1):
            if j % text_div == 0:
                plt.text(df.iloc[j, 0] + text_x_offset_end, df.iloc[j, 1] + text_y_offset_end,
                         word, color=cols_sub[j], fontsize=text_cex)

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()
