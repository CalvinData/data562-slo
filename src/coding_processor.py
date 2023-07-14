"""
This module creates a randomly sampled testset in CSV format for manual coding.
See main() for the details.
"""
import logging
from pathlib import Path
import requests
import fire
import pandas as pd
from src.settings import PTN_mention

logger = logging.getLogger(__name__)


def create_save_coding_set(output_filepath, data_frame, size, company_names, encoding):
    """This function creates coding sets and stores them in separate files."""
    coding_set = pd.DataFrame()

    for name, group in data_frame.groupby('company'):
        if (company_names is None) or (name in company_names):
            logging.info('\t{name} : {size} entries')
            coding_set = pd.concat(
                [coding_set, get_sample_tweets(group, size)],
                ignore_index=True
                )

    coding_columns = ['stance', 'confidence', 'value']
    data_columns = ['id', 'tweet_url', 'company', 'user_screen_name',
                    'tweet_norm', 'profile_norm']
    for column in coding_columns:
        coding_set[column] = '??'

    coding_set.to_csv(
        output_filepath,
        columns= coding_columns + data_columns,
        encoding=encoding,
        index=False
        )


def get_sample_tweets(group, size):
    """This function samples the given number (size) of tweets from the given
    group that satisfy the coding requirements.
    """
    group_coding_set = pd.DataFrame()
    current_ids = set()
    while get_size(group_coding_set) < size:
        # Sample one English tweet that isn't a bare retweet or a BOT tweet.
        sample_tweet = group[
            # Accept original tweets.
            (group['retweeted']) &
            # Accept empty or small hashtag lists.
            (group['hashtags'].str.len().isnull()) | (group['hashtags'].str.len() < 3)
            ].sample(1)
        sample_tweet['tweet_url'] = sample_tweet.id.apply(create_tweet_url)
        # Skip this tweet if it is deleted/invalid or repeated.
        if check_tweet_accessibility(sample_tweet['tweet_url'].values[0]) and \
                not sample_tweet['id'].values[0] in current_ids:
            # Add this usable tweet with coding text to the coding set.
            sample_tweet['tweet_to_code'] = sample_tweet['tweet_norm'].apply(
                remove_prepended_mentions
                )
            sample_tweet['stance'] = ''
            sample_tweet['confidence'] = ''
            group_coding_set = pd.concat(
                [group_coding_set, sample_tweet],
                ignore_index=True
                )
            current_ids.add(sample_tweet['id'].values[0])
            # Show progress in finding appropriate tweets.
            #     Useful if there are relatively few appropriate tweets
            # logging.info('\t\t%s', SLO_Dataset.get_size(group_coding_set))
    return group_coding_set


def get_size(data_frame):
    """This function gets the number of rows in the given dataframe."""
    return data_frame.shape[0]


def check_tweet_accessibility(tweet_url):
    """This function determines whether the given tweet is still accessible
    via the Twitter REST API.

    Arguments:
        tweet_url -- the URL for the tweet
    """
    response = requests.get(tweet_url, timeout=5)
    # Accessible tweets give HTTP 200 and include the screen name and tweet
    # ID in the URL. Inaccessible tweets can give 404 responses or redirect
    # to account/suspended.
    return response.status_code == 200 and \
        response.url.find('suspended') == -1


def remove_prepended_mentions(tweet):
    """Removes mentions that appear at the beginning of the tweet"""
    start = 0
    while True:
        match = PTN_mention.match(tweet, start)
        if match is None:
            break
        start = match.end()
    return tweet[start:]


def create_tweet_url(tweet_id):
    """This function constructs a URL referencing the full context of the
    given tweet.
    """
    return f'https://twitter.com/-/status/{tweet_id}'


def coding_processor(
    dataset_path='.',
    input_filename='dataset_norm.csv',
    output_filename='dataset_coding.csv',
    size=10,
    company_names=None,
    encoding='utf-8',
    logging_level=logging.INFO
    ):
    """This method selects a random set of tweets to code for each company of
    the given size. The tweets and a log of the creation process are stored
    in files using the given filename (.csv and .txt respectively). It
    assumes that the dataset has already been loaded. The columns are modified
    as follows:

    - A tweet_to_code column is added, which is the original tweet stripped of
        leading mentions.
    - A tweet_url column is added to give a direct Twitter URL for the tweet.
    - Empty columns are added for stance and confidence, to be filled in by
        human coders.

    The produced CSV file will include long integers that Excel doesn't handle
    by default. To solve the problem, import the .csv file as shown here:

    http://techsupport.matomy.com/Reports/29830196/How-to-present-long-numbers-correctly-in-Excel-CSV-file.htm

    Keyword arguments:
        dataset_path -- the system path from which to read the dataset
            (default: '.')
        input_filename -- the name of dataset input file
            (default: 'dataset_norm.csv')
        output_filename -- the name for the new coding output file
            (default: 'dataset_code.csv')
        size -- the number of coding set elements to sample for each company
            (default: 10)
        company_names -- a list of company names for which to collect samples
            (default: None -- collect for all companies)
        encoding -- the file encoding to use
            (default: 'utf-8')
        logging_level -- the level of logging to use
            (default: logging.INFO)
    """
    logging.basicConfig(
        level=logging_level,
        format='%(asctime)s %(levelname)s %(message)s',
        filename=__name__ + '.log',
        filemode='a'
        )
    logger.info('extracting manual coding dataset...')

    input_filepath = Path(dataset_path, input_filename)
    output_filepath = Path(dataset_path, output_filename)

    logging.info('\tloading dataset file: %s', input_filepath)
    # Use the python engine because it is more complete (but slower).
    data_frame = pd.read_csv(input_filepath, encoding=encoding, engine='python')

    logging.info('\tbuilding and saving the coding set to: %s', output_filepath)
    create_save_coding_set(output_filepath, data_frame, size, company_names, encoding)


if __name__ == '__main__':
    fire.Fire(coding_processor)

    # Example invocation:
    # python coding_processor.py --data_path=/media/hdd_2/slo/data/slo-tweets-20160101-20180304 --path='/media/hdd_2/slo/data/coding' --size=50 --company_names=['adani']
