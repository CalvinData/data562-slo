"""
This module implements the core pipeline of the SLO application
using a setup based on the original SLO tools.
"""
import logging
from pathlib import Path

from src.dataset_preprocessor import dataset_preprocessor
from src.dataset_normalizer import dataset_normalizer
from src.token_extractor import token_extractor

BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = Path(BASE_DIR, "data")
SRC_DIR = Path(BASE_DIR, "src")
FILENAME_BASE = 'dataset'
LOG_FILEPATH = Path(BASE_DIR, __name__ + '.log')

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    filename=LOG_FILEPATH,
    filemode='w'
    )


# Pipeline

# 1. Preprocess raw tweet dataset.
# Notes - to make this work, we needed to:
# - Convert the raw JSON data UTF-8 (using iconv -f latin-1 -t utf-8).
# - Remove some UTF-8 characters that messed up Polyglot/Cld2
#   (for details, see dataset_preprocessor#remove_bad_chars.).
dataset_preprocessor(
    dataset_path=DATA_DIR,
    input_filename=FILENAME_BASE + '.json',
    output_filename=FILENAME_BASE + '.csv'
    )

# 2. Normalize/tokenize pre-processed tweet dataset.
dataset_normalizer(
    dataset_path=DATA_DIR,
    input_filename=FILENAME_BASE + '.csv',
    output_filename=FILENAME_BASE + '_norm.csv'
    )

# 3. Create a tokens-only dataset for building word embeddings.
token_extractor(
    dataset_path=DATA_DIR,
    input_filename=FILENAME_BASE + '_norm.csv',
    output_filename=FILENAME_BASE + '_norm.txt'
    )



print(f'See the log output: {LOG_FILEPATH}')
# grep "\(processed [0-9]* records\|saved\)" __main__.log
