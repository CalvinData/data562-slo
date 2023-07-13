"""
This module builds a text file of tokens from the tweets and profile descriptions 
from a normalized dataset CSV file.
See main() for the details.
"""
import logging
import pandas as pd
from pathlib import Path
from fire import Fire

logger = logging.getLogger(__name__)


def token_extractor(
        dataset_path='.',
        dataset_filename='dataset_norm.csv',
        encoding='utf-8',
        logging_level=logging.INFO,
        logging_filename='token_extractor_log.txt',
        logging_mode='w'
        ):
    """This function extracts the raw text from the given tokenized dataset
    file. The text includes both the tweet text and the user profile
    description text. The input is assumed to have been tokenized by
    tweet_preprocessor.py. The output is written to path/filename.txt.

    Keyword Arguments:
        :param dataset_path: the root system path to the target/destination files
            (default: .)
        :param dataset_filename: the name of the dataset file
            (default: dataset_tok.csv)
        :param encoding: the file encoding
            (default: utf-8)
        logging_level -- the level of logging to use
            (default: logging.INFO)
        logging_filename -- the name of the log file
            (default: 'dataset_preprocessor_log.txt')
        logging_mode -- the mode to use when writing to the log file
            (default: 'w')
        logging_filename -- the name of the log file
            (default: 'token_extractor_log.txt')
        logging_mode -- the mode to use when writing to the log file
            (default: 'w')
    """
    logging.basicConfig(
        level=logging_level,
        format='%(message)s',
        filename=Path(dataset_path, logging_filename),
        filemode=logging_mode
        )
    logger.info('extracting tokens...')

    dataset_filepath = f'{dataset_path}/{dataset_filename}'
    logger.info('\tloading: %s', dataset_filepath)

    # Profiles are occasionally empty, so we need to drop the default NA handling.
    data_frame = pd.read_csv(dataset_filepath, keep_default_na=False)

    output_filename = f"{dataset_filename.split('.')[0]}_tokens.txt"
    output_filepath = f"{dataset_path}/{output_filename}"
    with open(output_filepath, 'w', encoding=encoding) as fout:
        # Dump unique tweet and profile texts (separately).
        fout.writelines([text for text in data_frame['tweet_norm'].unique() + '\n'])
        fout.writelines([text for text in data_frame['profile_norm'].unique() + '\n'])
        logger.info('saved to %s...', output_filepath)


if __name__ == '__main__':
    Fire(token_extractor)
