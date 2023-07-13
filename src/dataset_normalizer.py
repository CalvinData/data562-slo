"""
This module normalizes tweets for machine learning input.
See main() for the details.
"""
import csv
import html
import logging
from pathlib import Path
from fire import Fire
import pandas as pd

import src.settings

logger = logging.getLogger(__name__)


def preprocess_text(text: str) -> str:
    """This function normalizes the tweet field values."""
    try:
        text = html.unescape(text)
        text = src.settings.PTN_rt.sub('', text)
        text = src.settings.PTN_whitespace.sub(' ', text)
        text = src.settings.PTN_concatenated_url.sub(r'\1 http', text)

        # preserve Twitter specific tokens
        # username can contain year notations and elongations
        mentions = src.settings.PTN_mention.findall(text)
        text = src.settings.PTN_mention.sub(src.settings.SLO_MENTION_PLACEHOLDER, text)
        # URLs might be case sensitive
        urls = src.settings.PTN_url.findall(text)
        text = src.settings.PTN_url.sub(src.settings.SLO_URL_PLACEHOLDER, text)

        text = src.settings.PTN_elongation.sub(r'\1\1\1', text)
        text = text.lower()

        text = src.settings.PTN_year.sub('slo_year', text)
        text = src.settings.PTN_time.sub('slo_time', text)
        text = src.settings.PTN_cash.sub('slo_cash', text)
        text = src.settings.PTN_hash.sub('slo_hash', text)

        # put back Twitter specific tokens
        for url in urls:
            text = text.replace(src.settings.SLO_URL_PLACEHOLDER, url, 1)
        for mention in mentions:
            text = text.replace(src.settings.SLO_MENTION_PLACEHOLDER, mention, 1)

    except:
        logger.error('pre-precessing error on: %s; %s", text, type(text)')
        raise

    return text


def post_process_text(text: str) -> str:
    """Post-process an input text.

    This function modifies an input text as the precise input to word embedding tools

    - abstract mentions and URLs

    This does not touch hashtags and cashtags because treating them as different words will work for our task.
    """
    text = src.settings.PTN_mention.sub(r'slo_mention', text)
    text = src.settings.PTN_url.sub(r'slo_url', text)
    return text


def read_dataset(file_path: str, extension: str, encoding: str) -> pd.DataFrame:
    """This function reads the specified dataset, in whichever format."""
    logger.info('\tloading dataset file: %s', file_path)
    if extension == 'csv':
        # Adding na_filter here to ensure that empty strings are not converted to NaN.
        data_frame = pd.read_csv(file_path, encoding=encoding, engine='python', na_filter=False)
    elif extension == 'json':
        data_frame = pd.read_json(file_path)
    else:
        raise ValueError(f'file {file_path} not valid - only CSV and JSON accepted...')
    logger.info('\t\t%s items loaded', data_frame.shape[0])
    return data_frame


def save_datasets(data_frame: pd.DataFrame, filepath: str, separate_companies: bool) -> None:
    """This function saves the tokenized datasets, one for each company or
    one for all the companies combined.
    """
    data_frame.to_csv(filepath, index=False)
    logger.info('\t\tsaved %s items to %s', data_frame.shape[0], filepath)
    if separate_companies:
        for company_name, group in data_frame.groupby('company'):
            filepath = f'{filepath}-{company_name}_norm.csv'
            group.to_csv(filepath, index=False, quoting=csv.QUOTE_NONNUMERIC)
            logger.info('\t\tsaved %s items to %s', group.shape[0], filepath)


def fix_for_tagger(texts):
    """The CMU tokenizer/tagger doesn't handle empty ('') texts properly.
    Hack this by replacing them with 'PLACEHOLDER'. Tweets are never (?)
    empty, but user profile descriptions are frequently empty.
    """
    return [text if text != '' else 'slo_empty_text' for text in texts]


def dataset_normalizer(
        dataset_path: str='.',
        input_filename: str='dataset.csv',
        output_filename: str='dataset_norm.csv',
        tweet_column_name: str='text',
        profile_column_name: str='user_description',
        encoding: str='utf-8',
        separate_companies: bool=False,
        post_process: bool=False
        ) -> None:
    """This tool loads the preprocessed CSV-formatted tweets from the given
    filepath, normalizes the field values, and saves the results in a new
    filename (.csv). The normalization consists of:

    - RT sign (`RT @mention: `)
    - Shrink elongations:
        - letters -- waaaaaay -> waaay
        - signs   -- !!!!!!!! -> !!!
    - Downcase all letters
    - Replace newline (and tab) with a space
    - URLs:
        - Remove truncated URL fragments.
        - Add space between [.,?!] and 'http'.
    - Fix HTML escaped tokens
    - Abstract numeric values:
        - 1964  -> year
        - 16:32 -> time
        - $12   -> money
        - We don't abstract mentions or URLs by default because they may be more
            important for some tasks rather than others, e.g.: ML model input;
            creating word embeddings.

    Keyword Arguments
        :param dataset_path: the root system path to the target/destination files
            (default: .)
        :param dataset_filename: the name of the dataset file
            (default: dataset.csv)
        tweet_column_name:
            the column name of tweets in the input csv
            (default: 'text')
        profile_column_name:
            the column name of author profile description in the input csv
            (default: 'user_description')
        encoding:
            file character encoding
            (default: 'utf-8')
        separate_companies:
            if True, separate files grouped by company name are produced
            (default: False)
        post_process:
            if True, abstract mentions and URLs
            (default: False)
        logging_level
            the level of logging to use
            (default: logging.INFO)
        logging_filename -- the name of the log file
            (default: 'dataset_normalizer_log.txt')
        logging_mode -- the mode to use when writing to the log file
            (default: 'w')
    """
    logger.info('normalizing dataset...')

    _, extension = input_filename.split('.')
    input_filepath = Path(dataset_path, input_filename)
    output_filepath = Path(dataset_path, output_filename)

    data_frame = read_dataset(input_filepath, extension, encoding)

    logger.info('\tpre-processing tweet/profile texts...')
    tweets = data_frame[tweet_column_name].apply(preprocess_text)
    profiles = data_frame[profile_column_name].apply(preprocess_text)
    data_frame = data_frame.drop(
        columns=[tweet_column_name, profile_column_name]
        )

    if post_process:
        logger.info('\tpost-processing tweets...')
        tweets = tweets.apply(post_process_text)
        profiles = profiles.apply(post_process_text)

    logger.info('\tsaving normalized tweets and profiles:')
    data_frame['tweet_norm'] = tweets
    data_frame['profile_norm'] = profiles
    save_datasets(data_frame, output_filepath, separate_companies)


if __name__ == '__main__':
    Fire(dataset_normalizer)
