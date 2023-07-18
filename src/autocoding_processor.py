"""
This module creates a trainset with codes set automatically
based on hard-coded rules.
See main() for the details.
"""
import logging
from pathlib import Path
import re
import fire
import pandas as pd

from src.settings import \
    PTN_against, PTN_for, PTN_neutral_screen_names, \
    PTN_company_usernames, company_list

logger = logging.getLogger(__name__)


def get_size(df):
    """Get the number of rows in the given dataframe."""
    return df.shape[0]


def get_testset_ids_pattern(dataset_path, testset_filename, encoding):
    """Collect list of testset IDs that should not be included in the trainset, and
    return a compiled regex pattern for matching them.
    """

    # This shouldn't match anything, which we use as a default to ensure
    # that no tweets are excluded from the trainset.
    if testset_filename is None:
        return re.compile('$^')

    # Otherwise, create a pattern of IDS to exclude from the trainset.
    testset_filepath = Path(dataset_path, testset_filename)
    logger.info('\tloading testset file: %s', testset_filepath)
    df_testset = pd.read_csv(testset_filepath, encoding=encoding, engine='python')
    return re.compile('|'.join(pd.Series(df_testset['id']).apply(str)))


def create_tweet_sample(df_all, sample_size, code, company):
    """Sample the given number of tweets from the given dataframe for the given company"""
    if sample_size > 0:
        df_all = df_all.sample(sample_size)
    df_all['stance'] = code
    df_all['confidence'] = 'auto'
    df_all['company'] = company
    return df_all


def main(
    dataset_path='.',
    input_filename='dataset_norm.csv',
    output_filename='dataset_autocode.csv',
    testset_filename=None,
    encoding='utf-8',
    logging_level: int=logging.INFO,
    company_tweets=False
    ):
    """This function creates a auto-coded dataset using distance supervision,
    for Adani only, using simple hashtag rules. The three stance codings are
    balanced based on the number of "for" tweets.

    Keyword arguments:
        dataset_path:
            csv file path (default: 'dataset.csv')
        input_filename:
            input file name (default: 'dataset.csv')
        output_filename:
            output file name (default: 'dataset_autocode.csv')
        testset_filename:
            name of test (whose tweets should not be included in the trainset) 
            (default: None)
        encoding:
            file character encoding (default: 'utf-8')
        coding_path:
            path in which to dump the output file (default: '.')
        logging_level
            the level of logging to use (default: logging.INFO)
        company_tweets:
            whether to include tweets from company accounts (default: False)

    """
    logging.basicConfig(
        level=logging_level,
        format='%(asctime)s %(levelname)s %(message)s',
        filename=__name__ + '.log',
        filemode='a'
        )
    logger.info('auto-coding tweet trainset...')

    pd.options.mode.chained_assignment = None  # default='warn'

    input_filepath = Path(dataset_path, input_filename)
    output_filepath = Path(dataset_path, output_filename)

    testset_ids_pattern = get_testset_ids_pattern(dataset_path, testset_filename, encoding)

    df_all = pd.read_csv(input_filepath, encoding=encoding, engine='python')
    logger.info('\tloaded %s items from %s', get_size(df_all), input_filepath)

    df_combined = pd.DataFrame()
    for company in company_list:
        df_companies = df_all.loc[(df_all['company'].str.contains(company))]
        if testset_filename:
            df_companies = df_all.loc[
                (~df_all['id'].astype(str).str.match(testset_ids_pattern))
                ]
        logger.info('\t\t%s %s items loaded', get_size(df_companies), company)

        # Replace all semicolons to fix column displacement issues.
        df_companies['tweet_norm'] = df_companies['tweet_norm'].str.replace(";", "")
        df_companies['profile_norm'] = df_companies['profile_norm'].str.replace(";", "")

        # Annotate tweets with suspected stance values using rule patterns:
        # - For-stance tweets follow known positive patterns or come from the company itself.
        df_companies['auto_for'] = df_companies['tweet_norm'].str.contains(PTN_for[company])
        if company_tweets:
            df_companies['auto_for'] = pd.concat([
                df_companies['auto_for'],
                df_companies['user_screen_name'].str.match(PTN_company_usernames)
                ], ignore_index=True)
        # - Against-stance tweets follow known negative patterns.
        df_companies['auto_against'] = (
            df_companies['tweet_norm'].str.contains(PTN_against[company])
        )
        # - Neutral-stance tweets come from neutral accounts.
        df_companies['auto_neutral'] = (
            df_companies['user_screen_name'].str.match(PTN_neutral_screen_names)
        )

        # Collect tweets that are to be coded for each stance value.
        df_for = df_companies.loc[
            df_companies['auto_for'] & ~df_companies['auto_against']
            & ~df_companies['auto_neutral'] & ~df_companies['retweeted']
            ]
        df_against = df_companies.loc[
            ~df_companies['auto_for'] & df_companies['auto_against']
            & ~df_companies['auto_neutral'] & ~df_companies['retweeted']
            ]
        df_neutral = df_companies.loc[
            ~df_companies['auto_for'] & ~df_companies['auto_against']
            & df_companies['auto_neutral'] & ~df_companies['retweeted']
            ]

        min_sample_size = min(
            df_for.shape[0],
            df_against.shape[0],
            df_neutral.shape[0]
            )
        logger.info('\t\twill sample %s items per company', min_sample_size)


        # Get samples of for/against/neutral tweets.
        df_for = create_tweet_sample(df_for, min_sample_size, 'for', company)
        df_against = create_tweet_sample(df_for, min_sample_size, 'against', company)
        df_neutral = create_tweet_sample(df_for, min_sample_size, 'neutral', company)

        # ambiguous_df = df.loc[(df['auto_for'] & df['auto_against']) |
        #                       (~df['auto_for'] & ~df['auto_against'] & ~df['auto_neutral'])]

        # Remove the auto_* fields because they are useful only for computing the stance value.
        df_companies.drop(columns=['auto_for', 'auto_against', 'auto_neutral'])

        df_combined = pd.concat([df_combined, df_for, df_against, df_neutral], ignore_index=True)
        # logger.info(
        #     f'\tCompany: {company}\n\t\tfor: {get_size(df_for)} (out of {for_max_size})\n\t\tagainst: '
        #     f'{get_size(df_against)} (out of {against_max_size})\n\t\tneutral: {get_size(df_neutral)} '
        #     f'(out of {neutral_max_size})'
        #     )
        # index = output_filepath.find('.csv')
        # company_coding_filepath = output_filepath[:index] + "_" + company + output_filepath[index:]
        # pd.concat([df_for, df_against, df_neutral]).to_csv(company_coding_filepath)

    # Save the auto-coded items in one file.
    logger.info('\tstoring auto-coded dataset file: %s', output_filepath)
    df_combined.to_csv(output_filepath, index=False)


if __name__ == '__main__':
    fire.Fire(main)
    # Example invocation:
    # python autocoding_processor.py --dataset_filepath=/media/hdd_2/slo/stance/datasets/dataset.csv --coding_filepath=/media/hdd_2/slo/stance/coding --against_muliplier=1 --for_multiplier=1
