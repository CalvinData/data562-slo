"""
This module builds a text file of tokens from the tweet texts and profile
descriptions from a normalized dataset CSV file.
See main() for the details.
"""
import logging
from pathlib import Path
import pandas as pd
from fire import Fire

logger = logging.getLogger(__name__)


def token_extractor(
        dataset_path='.',
        input_filename='dataset_norm.csv',
        output_filename='dataset_norm_tokens.txt',
        encoding='utf-8'
        ):
    """This function extracts the raw text from the given tokenized dataset
    file. The text includes both the tweet text and the user profile
    description text. The input is assumed to have been normalized and tokenized.
    The output is written to a text file.

    Keyword Arguments:
        :param dataset_path: the root system path to the target/destination files
            (default: .)
        :param input_filename: the name of the dataset file
            (default: dataset_norm.csv)
        :param output_filename -- the name of the output file
            (default: dataset_norm_tokens.csv)
        :param encoding: the file encoding
            (default: utf-8)
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        filename=__name__ + '.log',
        filemode='a'
        )
    logger.info('extracting tokens...')

    input_filepath = Path(dataset_path, input_filename)
    logger.info('\tloading: %s', input_filepath)

    # Profiles are occasionally empty, so we need to drop the default NA handling.
    data_frame = pd.read_csv(input_filepath, keep_default_na=False)

    output_filepath = Path(dataset_path, output_filename)
    with open(output_filepath, 'w', encoding=encoding) as fout:
        # Dump unique tweet and profile texts (separately).
        fout.writelines([text for text in data_frame['tweet_norm'].unique() + '\n'])
        fout.writelines([text for text in data_frame['profile_norm'].unique() + '\n'])
        logger.info('saved to %s...', output_filepath)


if __name__ == '__main__':
    Fire(token_extractor)
