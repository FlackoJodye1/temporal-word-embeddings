import numpy as np
import pandas
import pandas as pd
from glob import glob
from pathlib import Path
from tqdm.notebook import tqdm

import sys

sys.path.append('../')
from src.packages.TPPMI.tppmi_model import TPPMIModel


# -------------------------------------------------------------------------------- #
# ---------------------- Creation of Testword-Dicts ------------------------------ #
# -------------------------------------------------------------------------------- #


def create_test_case_dict_cade(test_cases: np.ndarray, models: dict) -> dict:
    test_case_dict = dict()
    counter = 0
    for test_case in test_cases:
        word, year = test_case.split("-")
        ground_model = models[f"model_{year}"]
        if word in ground_model.wv.key_to_index:
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
        if word in model.wv.key_to_index:
            test_case_dict[test_case] = model.wv.get_vector(word)
        else:
            counter = counter + 1
    print(f"{counter} Testcases are not in the vocab of the model")
    return test_case_dict


def get_similarities_of_models_2(model_dict: dict, test_word_dict: dict, top_n=10) -> dict:
    similarities = dict()
    for test_word in tqdm(test_word_dict.items()):
        similarities[test_word[0]] = dict()
        for model in model_dict.items():
            similarities[test_word[0]][model[0].split("_")[1]] = model[1].wv.similar_by_vector(test_word[1], topn=top_n)
    return similarities


# -------------------------------------------------------------------------------- #
# ------------------- Calculation of Similarity Scores --------------------------- #
# -------------------------------------------------------------------------------- #

def get_similarites_of_models_tppmi(model: TPPMIModel,
                                    test_word_dict: dict, entity_list= None, label= None, top_n=10, filter=False) -> dict:
    similarities = dict()
    for word, vector in tqdm(test_word_dict.items()):
        if filter:
            similarities_for_one_word = model.most_similar_words_by_vector(vector,
                                                                  top_n=top_n, filter=filter, entity_list=entity_list, label=label)
        else:
            similarities_for_one_word = model.most_similar_words_by_vector(vector, top_n=top_n)
        similarities[word] = similarities_for_one_word
    return similarities


def filter_similarities(similarity_list, label, entity_list, top_n):
    new_value_list = []
    for word, similarity in similarity_list:
        word_label = entity_list['entity'].loc[entity_list['token'] == word].iloc[0] if not entity_list.loc[entity_list['token'] == word].empty else None
        if word_label == label:
            new_value_list.append((word, similarity))  # Assuming similarity scores need to be preserved
        if len(new_value_list) >= top_n:
            break
    return new_value_list


def get_similarities_of_models(model_dict: dict, test_word_dict: dict, entity_list: pd.DataFrame, label: str, top_n=10) -> dict:
    similarities = dict()
    for test_word, vector in tqdm(test_word_dict.items()):
        similarities[test_word] = dict()
        for model_name, model in model_dict.items():
            year = model_name.split("_")[1]  # Extract the year from the model name
            raw_similarities = model.wv.similar_by_vector(vector, topn=(2 * top_n))
            filtered_similarities = filter_similarities(raw_similarities, label, entity_list, top_n)

            # Check if filtered results meet the required top_n, if not, increment and fetch more
            increment = 0
            while len(filtered_similarities) < top_n:
                increment += 10
                print(test_word)
                print(f"incremented by: {increment}")
                additional_similarities = model.wv.similar_by_vector(vector, topn=top_n + increment)
                filtered_similarities = filter_similarities(additional_similarities, label, entity_list, top_n)
            similarities[test_word][year] = filtered_similarities
    return similarities


def get_similarities_of_models_old(model_dict: dict, test_word_dict: dict, top_n=10) -> dict:
    similarities = dict()
    for test_word in tqdm(test_word_dict.items()):
        similarities[test_word[0]] = dict()
        for model in model_dict.items():
            similarities[test_word[0]][model[0].split("_")[1]] = model[1].wv.similar_by_vector(test_word[1], topn=20)
    return similarities


def get_similarities_of_models_static(model, test_word_dict: dict, entity_list: pd.DataFrame, top_n=5,
                                      filter=False) -> dict:
    similarities = {}
    for test_word, vector in tqdm(test_word_dict.items()):
        raw_similarities = model.wv.similar_by_vector(vector, topn=(2 * top_n))
        if filter:
            filtered_similarities = filter_similarities_static(raw_similarities, label="PERSON",
                                                               entity_list=entity_list,
                                                               top_n=top_n)
        else:
            filtered_similarities = raw_similarities
        # Ensure we retrieve enough similar words
        increment = 0
        while len(filtered_similarities) < top_n:
            increment += 10  # Increment by a reasonable amount to avoid too many iterations
            print(test_word)
            print(f"incremented by: {increment}")
            more_similarities = model.wv.similar_by_vector(vector, topn=top_n + increment)
            filtered_similarities = filter_similarities_static(more_similarities, label="PERSON",
                                                               entity_list=entity_list, top_n=top_n)
        similarities[test_word] = filtered_similarities
    return similarities


def filter_similarities_static(similarity_list, label: str, entity_list: pd.DataFrame, top_n: int):
    new_value_list = []
    for word, score in similarity_list:
        match = entity_list["entity"].loc[entity_list.token == word]
        if not match.empty:
            word_label = entity_list["entity"].loc[entity_list.token == word].iloc[0]
        else:
            continue
        if label == word_label:
            new_value_list.append((word, score))
        if len(new_value_list) >= top_n:
            break
    return new_value_list


# -------------------------------------------------------------------------------- #
# ------------------- Calculation of Rank/Score metrics -------------------------- #
# -------------------------------------------------------------------------------- #


def get_quick_mrr(test_key: str, similarities: dict, test_set: pandas.DataFrame) -> float:
    test_data_for_key = similarities[test_key]
    ranks = []

    for key, values in test_data_for_key.items():
        retrieved_list = [value[0] for value in values]
        equivalent = test_set[test_set.year == key]["name"].values
        rank = calculate_reciprocal_rank(retrieved_list, equivalent)
        ranks.append(rank)

    if ranks:  # Ensure division by 0 does not occur
        mean_rank = sum(ranks) / len(ranks)
    else:
        mean_rank = 0

    return round(mean_rank, 3)


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


def filter_similarities_old(similaritiy_dict, entity_list: pd.DataFrame, min_length=5):
    for testcase, similarities in tqdm(similaritiy_dict.items()):
        label = ""
        token = testcase.split("-")[0]
        try:
            label = entity_list["entity"].loc[entity_list.token == token].iloc[0]
        except IndexError:
            print(f"Entity-Type of {token} can not be determined")
        print(f"{token} - {label}")
        for key, value_list in similarities.items():
            new_value_list = []
            for entry in value_list:
                word = entry[0]
                try:
                    word_label = entity_list["entity"].loc[entity_list.token == word].iloc[0]
                except IndexError:
                    continue
                if label == word_label:
                    new_value_list.append(word)
            if len(new_value_list) < min_length:
                raise Exception(f"Min-Length requirement of {min_length} for the list of neighbors is not satisfied "
                                f"for testcase {testcase}")
            else:
                # slices the list up to min_length
                similaritiy_dict[testcase][key] = new_value_list[:min_length]


def load_score_tables(data_dir):
    """
    Load all CSV score tables from a specified directory and return a dictionary of DataFrames.
    The dictionary keys will be sorted alphabetically.

    Args:
    data_dir (Path or str): The directory path where the score tables are saved as CSV files.

    Returns:
    dict: A dictionary of pandas DataFrames, sorted by the model names.
    """
    # Use glob to find all CSV files in the directory
    csv_files = glob(str(data_dir / "*.csv"))

    # Create a dictionary with model names and their DataFrames, then sort by model name
    score_tables = {
        Path(csv_file).stem: pd.read_csv(csv_file, index_col=0)
        for csv_file in csv_files
    }

    # Sort the dictionary by keys (model names) and return it
    sorted_score_tables = dict(sorted(score_tables.items(), key=lambda x: x[0].lower()))

    return sorted_score_tables
