"""
This module normalizes tweets for machine learning input.
See main() for the details.
"""
import csv
import html
import logging

from fire import Fire
import pandas as pd

import settings
# The CMU Tweet Tagger import and code (below) are commented out for the moment.
#    The tagger's a big Java utility and it's not clear that it will be useful.
#from vendor import CMUTweetTagger as Tagger

logger = logging.getLogger(__name__)


def preprocess_text(text: str) -> str:
    """This function normalizes the tweet field values."""
    try:
        text = html.unescape(text)
        text = settings.PTN_rt.sub('', text)
        text = settings.PTN_whitespace.sub(' ', text)
        text = settings.PTN_concatenated_url.sub(r'\1 http', text)

        # preserve Twitter specific tokens
        # username can contain year notations and elongations
        mentions = settings.PTN_mention.findall(text)
        text = settings.PTN_mention.sub(settings.SLO_MENTION_PLACEHOLDER, text)
        # URLs might be case sensitive
        urls = settings.PTN_url.findall(text)
        text = settings.PTN_url.sub(settings.SLO_URL_PLACEHOLDER, text)

        text = settings.PTN_elongation.sub(r'\1\1\1', text)
        text = text.lower()

        text = settings.PTN_year.sub('slo_year', text)
        text = settings.PTN_time.sub('slo_time', text)
        text = settings.PTN_cash.sub('slo_cash', text)
        text = settings.PTN_hash.sub('slo_hash', text)

        # put back Twitter specific tokens
        for url in urls:
            text = text.replace(settings.SLO_URL_PLACEHOLDER, url, 1)
        for mention in mentions:
            text = text.replace(settings.SLO_MENTION_PLACEHOLDER, mention, 1)

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
    text = settings.PTN_mention.sub(r'slo_mention', text)
    text = settings.PTN_url.sub(r'slo_url', text)
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
    csv_filename = f'{filepath}_norm.csv'
    data_frame.to_csv(csv_filename, index=False)
    logger.info('\t\t%s - %s items', csv_filename, data_frame.shape[0])
    if separate_companies:
        for company_name, group in data_frame.groupby('company'):
            csv_filename = f'{filepath}-{company_name}_norm.csv'
            group.to_csv(csv_filename, index=False, quoting=csv.QUOTE_NONNUMERIC)
            logger.info('\t\t%s - %s items', csv_filename, group.shape[0])


def fix_for_tagger(texts):
    """The CMU tokenizer/tagger doesn't handle empty ('') texts properly.
    Hack this by replacing them with 'PLACEHOLDER'. Tweets are never (?)
    empty, but user profile descriptions are frequently empty.
    """
    return [text if text != '' else 'slo_empty_text' for text in texts]


def main(
    csv_data_filepath: str='dataset.csv',
    tweet_column_name: str='text',
    profile_column_name='user_description',
    encoding: str='utf-8',
    separate_companies: bool=False,
    post_process: bool=True,
    logging_level: int=logging.INFO
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

    # Args

    csv_data_filepath:
        CSV or JSON file path
        (default: 'dataset.csv')
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
        (default: True)
    logging_level
        the level of logging to use
        (default: logging.INFO)
    """
    logging.basicConfig(level=logging_level, format='%(message)s')
    logger.info('normalizing tweets')

    filename, extension = csv_data_filepath.split('.')
    data_frame = read_dataset(csv_data_filepath, extension, encoding)

    logger.info('\tpre-processing tweets and profile descriptions...')
    tweets = data_frame[tweet_column_name].apply(preprocess_text)
    profiles = data_frame[profile_column_name].apply(preprocess_text)
    data_frame = data_frame.drop(
        columns=[tweet_column_name, profile_column_name]
        )

    # logger.info(f'\tparsing/tagging tweets...')
    # tagged = Tagger.runtagger_parse(fix_for_tagger(tweets))
    # tweets = pd.Series([
    #     ' '.join([w for w, pos, p in row]) for row in tagged
    # ])
    # tagged = Tagger.runtagger_parse(fix_for_tagger(profiles))
    # profiles = pd.Series([
    #     ' '.join([w for w, pos, p in row]) for row in tagged
    # ])

    if post_process:
        logger.info('\tpost-processing tweets...')
        tweets = tweets.apply(post_process_text)
        profiles = profiles.apply(post_process_text)

    logger.info('\tsaving normalized tweets and profiles:')
    data_frame['tweet_norm'] = tweets
    data_frame['profile_norm'] = profiles
    save_datasets(data_frame, filename, separate_companies)


if __name__ == '__main__':
    Fire(main)
    # Example invocation (CSIRO, 2018):
    # python dataset_normalizer.py --csv_data_filepath=/media/hdd_2/slo/stance/datasets/dataset.csv --post_process
