import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

import sys

sys.path.append('../')
from src.packages.TPPMI.tppmi_model import TPPMIModel


def create_test_case_dict_cade(test_cases: np.ndarray, models: dict) -> dict:
    test_case_dict = dict()
    counter = 0
    for test_case in test_cases:
        word, year = test_case.split("-")
        ground_model = models[f"model_{year}"]
        if word in ground_model.wv.vocab:
            test_case_dict[test_case] = ground_model.wv.get_vector(word)
        else:
            counter = counter + 1
    print(f"{counter} Testcases are not in the vocab of the model(s)")
    return test_case_dict


def create_test_case_dict_static(model, test_cases: np.ndarray) -> dict:
    test_case_dict = dict()
    counter = 0
    for test_case in test_cases:
        word, year = test_case.split("-")
        if word in model.wv.vocab:
            test_case_dict[test_case] = model.wv.get_vector(word)
        else:
            counter = counter + 1
    print(f"{counter} Testcases are not in the vocab of the model")
    return test_case_dict


def get_similarities_of_models(model_dict: dict, test_word_dict: dict) -> dict:
    similarities = dict()
    for test_word in tqdm(test_word_dict.items()):
        similarities[test_word[0]] = dict()
        for model in model_dict.items():
            similarities[test_word[0]][model[0].split("_")[1]] = model[1].wv.similar_by_vector(test_word[1])
    return similarities


def get_similarities_of_models_static(model, test_word_dict: dict) -> dict:
    similarities = dict()
    for test_word in tqdm(test_word_dict.items()):
        similarities[test_word[0]] = model.wv.similar_by_vector(test_word[1])
    return similarities


def calculate_mean_rank(test_key: str, testcase: dict, test_data: pd.DataFrame, metric="MRR", k=10) -> float:
    test_data_for_key = test_data[test_data["truth"] == test_key]
    ranks = []

    for key, value in testcase.items():
        test_data_for_year = test_data_for_key[test_data_for_key["equivalent"].str.endswith(key)]
        word_list = [item[0] for item in value]

        if len(test_data_for_year) == 0:
            continue  # Skip if no data for year, as there's nothing to rank
        target_word = test_data_for_year["equivalent"].iloc[0].split("-")[0]
        if metric == "MRR":
            rank = calculate_reciprocal_rank(word_list, target_word)
        else:
            rank = calculate_precision_at_k(word_list, target_word, k)

        ranks.append(rank)

    if ranks:  # Ensure division by 0 does not occur
        mean_rank = sum(ranks) / len(ranks)
    else:
        mean_rank = 0

    return mean_rank


def calculate_reciprocal_rank(test_list: list, test_word: str) -> float:
    """
    Calculate the reciprocal rank for a given test word in a list of strings.

    Parameters:
    test_list (list of str): The list of strings to search through.
    test_word (str): The correct answer to find in the test_list.
    Returns:
    float: The reciprocal rank of the test_word in test_list, or 0 if not found.
    """
    try:
        rank = test_list.index(test_word) + 1  # Adding 1 because index is 0-based and rank is 1-based
        return 1.0 / rank
    except ValueError:
        return 0.0  # test_word not found in test_list


def calculate_precision_at_k(test_list: list, test_word: str, k: int) -> int:
    """
    Calculate the precision at K for a given test word in a list of strings.

    Parameters:
    test_list (list of str): The list of strings to search through, assumed to be ordered by relevance.
    test_word (str): The correct answer to find in the test_list.
    k (int): The number of top items to consider for calculating precision.

    Returns:
    int: The precision at K for the test_word in test_list.
         If the target word is among these K words, then the Precision@K for test i
         (denoted P@K[i]) is 1; else, it is 0
    """
    if k <= 0:
        raise ValueError("k must be a positive integer")

    # Take the top K elements from the list
    top_k = test_list[:k]

    # Check if the test_word is within the top K elements
    if test_word in top_k:
        return 1
    else:
        return 0


def calculate_rank_metric(similarities: dict, test_data: pd.DataFrame, metric="MRR", k=10) -> float:
    ranks = []
    for key, value in similarities.items():
        rank = calculate_mean_rank(key, value, test_data, metric, k)

        ranks.append(rank)

    if ranks:  # Ensure division by 0 does not occur
        mean_rank = sum(ranks) / len(ranks)
    else:
        mean_rank = 0

    return mean_rank


def calculate_rank_metric_static(similarities: dict, test_data: pd.DataFrame, metric="MRR", k=10) -> float:
    ranks = []
    counter = 0

    for key, value in similarities.items():
        test_data_for_key = test_data[test_data["truth"] == key]
        word_list = [item[0] for item in value]
        for _, test_case in test_data_for_key.iterrows():
            target_word = test_case["equivalent"].split("-")[0]
            if metric == "MRR":
                rank = calculate_reciprocal_rank(word_list, target_word)
            else:
                rank = calculate_precision_at_k(word_list, target_word, k)
            ranks.append(rank)
            counter = counter + 1

    if ranks:  # Ensure division by 0 does not occur
        mean_rank = sum(ranks) / len(ranks)
    else:
        mean_rank = 0

    return mean_rank


def create_test_case_dict_tppmi(model: TPPMIModel, test_cases: np.ndarray) -> dict:
    test_case_dict = dict()
    counter = 0
    for test_case in test_cases:
        word, year = test_case.split("-")
        if model.is_in_vocab_of_timestep(word, year):
            df = model.get_tppmi([word])
            df = df[next(iter(df))]
            test_case_dict[test_case] = df.loc[f"{word}_{year}"].to_numpy()
        else:
            counter = counter + 1
    print(f"{counter} Testcases are not in the vocab of the model")

    return test_case_dict


def get_similarites_of_models_tppmi(model: TPPMIModel, test_word_dict: dict) -> dict:
    similarities = dict()
    for word, vector in tqdm(test_word_dict.items()):
        similarities[word] = model.most_similar_words_by_vector(vector)
    return similarities
