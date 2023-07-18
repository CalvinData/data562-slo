"""
Utility modules of data handling on machine learning.
"""
import csv
import logging
from typing import Dict, List, Tuple
import numpy as np

from src.settings import PTN_against, PTN_for

Dsets = Dict[str, np.ndarray]
logger = logging.getLogger(__name__)


def get_x(row: dict, auto_tagged: bool, profile: bool) -> str:
    """Adds metadata along with main tweet text to include:
        - target company
        - tweet text
        - profile text
    """
    features = list()

    # The order of these features must match the order used in split_x_value().
    features.append(row['company'])
    features.append(row['tweet_norm'])
    if profile:
        features.append(row['profile_norm'])
    output = '\t'.join(features)

    # If the data was auto-coded, remove auto-tagging hashtags.
    if auto_tagged:
        output = PTN_for[row['company']].sub('', output)
        output = PTN_against[row['company']].sub('', output)

    return output


def dic_list2array(lists: Dict[str, list]) -> Dsets:
    """Coverts a dictionary of lists to a dictionary of numpy arrays"""
    return {target: np.array(lst) for target, lst in lists.items()}


def split_x_value(x, profile_flag):
    """This function returns the company target and tweet text strings. If
    profile is true, include the profile text as well, otherwise return None
    as the profile text. The order of the split must match the order in get_x();
    the order of the return values is: target, tweet, profile.
    """
    if profile_flag:
        target, tweet, profile = x.split('\t')
    else:
        target, tweet = x.split('\t')
        profile = ''
    return target, tweet, profile


# def load_dataset(dataset_filepath: str, labels: list, encoding: str) -> Tuple[Dsets, Dsets]:
#     """Loads the specified dataset, with no splitting of train/test sets"""

#     x_lists: Dict[str, List[str]] = {}
#     y_lists: Dict[str, List[int]] = {}

#     # Detect whether the dataset is auto-coded.
#     auto_tagged = 'auto' in str(dataset_filepath)
#     if auto_tagged:
#         logger.info('\t\tdetected auto-coded data - removing query hashtags from tweet texts...')

#     with open(dataset_filepath, encoding=encoding) as fin:
#         reader = csv.DictReader(fin)
#         for row in reader:
#             x = get_x(row, auto_tagged=auto_tagged)
#             y = labels.index(row['stance'].strip())

#             x_lists.setdefault(row['company'], []).append(x)
#             y_lists.setdefault(row['company'], []).append(y)

#     x_arrays = dic_list2array(x_lists)
#     y_arrays = dic_list2array(y_lists)

#     logger.info('\tloaded %s records from %s', len(x_arrays), dataset_filepath)

#     return x_arrays, y_arrays

def load_dataset(dataset_filepath, labels, encoding='utf-8', profile=True):
    """Load a SLO dataset and return X and Y.

    Mostly same as `load_data` but doesn't split the target datasets and
    doesn't remove query hashtags.
    """
    x_items = []
    y_items = []

    # Detect whether the dataset is auto-coded.
    auto_tagged = 'auto' in str(dataset_filepath)
    if auto_tagged:
        logger.info('\t\tdetected auto-coded data - removing query hashtags from tweet texts...')

    with open(dataset_filepath, encoding=encoding) as f:

        for row in csv.DictReader(f):
            x = get_x(row, auto_tagged=auto_tagged, profile=profile)
            y = labels.index(row['stance'].strip())

            x_items.append(x)
            y_items.append(y)

    return np.asarray(x_items), np.asarray(y_items)


def translate_predicted(y_predicted, labels):
    """Converts the predicted codes to their corresponding label."""
    return [labels[x] for x in y_predicted]


def set_labels(labels):
    """Sets the labels for the dataset to the default labels if not specified.
    Set to negative = 0, positive = 1 due to the calculation for macroF measure.
    """
    if labels is None:
        labels = ['against', 'for', 'neutral', 'na']
    return labels
