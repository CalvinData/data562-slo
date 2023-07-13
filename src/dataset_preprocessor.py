"""
This module builds a dataset file from a JSON file of raw tweets.
See main() for the details.
"""
import os
import csv
import json
import re
import logging
from pathlib import Path
from fire import Fire
import pandas as pd
from polyglot.text import Text

from src.settings import PTN_rt, PTN_companies, RETWEET_START, REGEX_BAD_CHARS

logger = logging.getLogger(__name__)

# Count irrelevant tweets.
__unknown_company_count_global__ = 0
__non_english_count_global__ = 0

def create_dataset(
        input_filepath,
        output_filepath,
        encoding,
        drop_irrelevant_tweets,
        keep_retweets
        ):
    """This function rebuilds a dataset from the given raw JSON file."""
    logger.info('\tloading raw tweets from %s', input_filepath)

    # Load/save the file in chunks.
    count = 0
    include_header = True
    for df_chunk in pd.read_json(
        input_filepath,
        orient='records',
        lines=True,
        chunksize=50000,
        encoding=encoding,
        ):

        # Create/update/infer fields.
        df_chunk['retweeted'] = df_chunk.apply(compute_retweet, axis=1)
        df_chunk['text'] = \
            df_chunk.apply(compute_full_text, axis=1)
        df_chunk['lang_polyglot'] = \
            df_chunk.apply(update_language, axis=1)
        df_chunk[['user_screen_name', 'user_description']] = \
            df_chunk.apply(compute_user_series, axis=1)
        df_chunk['hashtags'] = \
            df_chunk.apply(compute_hashtags, axis=1)
        df_chunk['company'] = df_chunk.apply(compute_company, axis=1)

        # Remove irrelevant tweets (non-English or unknown-company).
        if drop_irrelevant_tweets:
            logger.info('\t\tdropping non-English/unknown-company tweets...')
            df_chunk = df_chunk[
                (df_chunk['company'] != '') &
                (
                    df_chunk['lang'].str.startswith('en')
                    | df_chunk['lang_polyglot'].str.startswith('en')
                )
                ]

        # Remove retweets.
        if not keep_retweets:
            logger.info('\t\tdropping retweets...')
            df_chunk = df_chunk[~df_chunk['retweeted']]

        # Write each chuck to the combined dataset file.
        required_fields = [
            'id',
            'created_at',
            'lang',
            'lang_polyglot',
            'retweeted',
            'hashtags',
            'company',
            'text',
            'user_screen_name',
            'user_description'
            ]
        df_chunk[required_fields].to_csv(
            output_filepath,
            index=False,
            quoting=csv.QUOTE_NONNUMERIC,
            mode='a',
            header=include_header,
            )

        # Print a progress message.
        count += get_size(df_chunk)
        # Only include the header once, at the top of the file.
        include_header = False
        logger.info('\t\tprocessed %s records...', count)

    # Adding na_filter here to ensure that empty strings are not converted to NaN.
    df_full = pd.read_csv(output_filepath, na_filter=False)
    df_full.drop_duplicates(inplace=True)
    df_full.to_csv(output_filepath, index=False, header=True, quoting=csv.QUOTE_NONNUMERIC)
    logger.info(
        '\tsaved the dataset to %s' +
        '\n\t\tunknown company count: %s' +
        '\n\t\tnon-English count: %s',
        output_filepath, __unknown_company_count_global__, __non_english_count_global__
        )


def compute_retweet(row):
    """This function determines if a tweet is a retweet."""
    return row['full_text'].startswith(RETWEET_START)


def compute_full_text(row):
    """This function creates the full text, either from the tweet itself or,
    if the tweet is a retweet (RT) that has been truncated (... at the end),
    by pasting the retweet header onto the original tweet text found in the
    retweet information.
    """
    full_text = row['full_text']

    # If needed, reconstruct the full tweet text from the original text,
    # leaving the retweet header intact.
    if full_text.startswith(RETWEET_START) \
            and full_text.endswith('\u2026') \
            and not pd.isnull(row['retweeted_status']):
        text_header = PTN_rt.search(row['full_text']).group()
        retweet_full_text = pd.read_json(
            json.dumps(row['retweeted_status']['full_text']),
            typ='series'
            )[0]
        full_text = f'{text_header}{retweet_full_text}'

    return clean_text(full_text)


def remove_bad_chars(text):
    """This module implements a hack that removes the types of UTF-8 characters
    that mess up PolyGlot/cld2. See:
    - https://github.com/aboSamoor/pycld2/issues/53
    - https://github.com/aboSamoor/polyglot/issues/71#issuecomment-707997790
    """
    return REGEX_BAD_CHARS.sub("", text)


def update_language(row):
    """This function computes an alternate language code for the given
    tweet using TextBlob, a more reliable language coder.
    """
    global __non_english_count_global__
    if row['lang'].startswith('en'):
        # Leave English codes (e.g., en, en-gb) unchanged.
        return row['lang']
    else:
        # Compute alternate code for non-English tweets, many of which are
        # actually in English as well.
        lang2 = Text(remove_bad_chars(row['full_text'])).language.code
        if not lang2.startswith('en'):
            __non_english_count_global__ += 1
            logger.warning(
                "\t\t\tnon-English tweet (will be dropped): " +
                "\n\t\t\t\tid: %s" +
                "\n\t\t\t\ttweet: %s" +
                "\n\t\t\t\tLanguage tags: %s - %s",
                row['id'], row['text'], row['lang'], lang2
                )
        return lang2


def compute_user_series(row):
    """This function grabs the user name and profile description from the
    nested user JSON structure.
    """
    user_series =  pd.read_json(json.dumps(row['user']), typ='series')
    user_series['description'] = clean_text(user_series['description'])
    return user_series[['screen_name', 'description']]


def compute_hashtags(row):
    """This function grabs the list of hashtags from the nested entities
    JSON structure.
    """
    entity_series =  pd.read_json(json.dumps(row['entities']), typ='series')
    hashtags = list(map(lambda entry: entry['text'], entity_series['hashtags']))
    return ','.join(hashtags)


def compute_company(row):
    """This function identifies the target company from the tweet text, issuing
    a warning for unrecognized texts. It assumes that the full, un-truncated
    tweet text has already been constructed (see compute_full_text()).
    """
    global __unknown_company_count_global__
    associated_company = []

    # Identify the target company using known patterns in the tweet text.
    tweet = row['text'].lower()
    author = row['user_screen_name'].lower()
    for company_pattern in PTN_companies:
        if re.compile(author).fullmatch(company_pattern[2]):
            associated_company.append(company_pattern[0])
            break
        if company_pattern[1].search(tweet):
            associated_company.append(company_pattern[0])

    if len(associated_company) > 0:
        return '|'.join(associated_company)

    # No company pattern applies, so it's unclear how this tweet was selected.
    __unknown_company_count_global__ += 1
    logger.warning(
        "\t\t\tunrecognized company (will be dropped): " +
        "\n\t\t\t\tid: %s" +
        "\n\t\t\t\ttweet: %s" +
        "\n\t\t\t\thashtags: %s",
        row['id'], row['text'], row['hashtags']
        )
    return ''


def get_size(data_frame):
    """Get the number of rows in the given dataframe."""
    return data_frame.shape[0]


def clean_text(text):
    """Do simple text cleanup for the data processing files."""
    return text.replace('\n', ' ').replace('\r', ' ')


def remove_filepath_if_exists(filepath):
    """Delete the given file if it exists."""
    if os.path.isfile(filepath):
        logger.info('\tdeleting existing file: %s', filepath)
        os.remove(filepath)


def create_separate_company_datasets(input_filepath, output_path, filename_base):
    """Read the given full/combined dataset file and create/save
    company-specific groups.
    """
    logger.info('\tsplitting dataset into company-specific datasets...')
    data_frame = pd.read_csv(input_filepath, encoding='utf-8', engine='python')
    for company_name, group in data_frame.groupby(['company']):
        group.to_csv(
            output_path / f'{filename_base}-{company_name}.csv',
            index=False)


def dataset_preprocessor(
        dataset_path='.',
        input_filename='dataset.json',
        output_filename='dataset.csv',
        encoding='utf-8',
        drop_irrelevant_tweets=True,
        keep_retweets=True,
        add_company_datasets=False
        ):
    """This tool loads the raw JSON-formatted tweets from the given
    filepath, does some general updates to the dataset items and saves
    the results in filename (.csv). The columns are modified as
    follows:

    - The tweet text is modified to remove newlines (\\n, \\r).
    - Columns are added for the following:
        - The company referred to by the tweet - If no known company is identified,
            the tweet is dropped.
        - The language in which the tweet is written - The Twitter feed has a language
            column, lang, but it's often wrong. So, we use Polyglot to add a column,
            lang_polyglot, that we use as a second, often more accurate, opinion,
            keeping the tweet if either language field is set to English.
        - The full text of the tweet - We either use the twitter text column or we
            construct the text from the re-tweet information.
        - Hashtags - We add a column with the comma-separated list of hashtags.
        - Author information - We include the author's screen name and add their
            profile description.

    Keyword Arguments:
        dataset_path -- the system path from which to load the raw JSON files
            (default='.')
        dataset_filename -- the name of the input file to read
            (default='raw_dataset.json')
        encoding -- the file encoding to use
            (default: 'utf-8')
        drop_irrelevant_tweets -- whether to drop tweets that are either:
            not in English or talk about an unknown company
            (default: True)
        keep_retweets -- whether to keep retweeted tweets
            (default: True)
        add_company_datasets -- whether to add company-specific datasets
            (default: False)
        logging_level -- the level of logging to use
            (default: logging.INFO)
        logging_filename -- the name of the log file
            (default: 'dataset_preprocessor_log.txt')
        logging_mode -- the mode to use when writing to the log file
            (default: 'w')
    """
    logger.info('pre-processing dataset...')

    input_filepath = Path(dataset_path, input_filename)
    output_filepath = Path(dataset_path, output_filename)
    remove_filepath_if_exists(output_filepath)

    create_dataset(
        input_filepath,
        output_filepath,
        encoding,
        drop_irrelevant_tweets,
        keep_retweets
        )

    if add_company_datasets:
        create_separate_company_datasets(
            input_filepath,
            dataset_path,
            output_filepath.stem
            )


if __name__ == '__main__':
    Fire(dataset_preprocessor)
