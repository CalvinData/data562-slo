"""
This module implements the core operations of the SLO application
using a setup based on the original SLO tools and MadeWithML's template.
"""
from timeit import default_timer as timer
from datetime import timedelta
from pathlib import Path

from src.dataset_preprocessor import dataset_preprocessor
from src.dataset_normalizer import dataset_normalizer
from src.token_extractor import token_extractor

# Directories
BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = Path(BASE_DIR, "data")
SRC_DIR = Path(BASE_DIR, "src")

# Preprocess raw tweet dataset.
# Notes - to make this work, we needed to:
# - Convert the raw JSON data UTF-8 (using iconv -f latin-1 -t utf-8).
# - Remove some UTF-8 characters that messed up Polyglot/Cld2
#   (for details, see dataset_preprocessor#remove_bad_chars.).
start = timer()
dataset_preprocessor(
    dataset_path=DATA_DIR,
    dataset_filename='test.json',
    logging_filename='test.json.log'
    )
end = timer()
print(f"dataset_preprocessor: {timedelta(seconds=end-start)}")

# Normalize pre-processed tweet dataset.
start = timer()
dataset_normalizer(
    dataset_path=DATA_DIR,
    dataset_filename='dataset.csv',
    logging_filename='dataset.csv.log'
    )
end = timer()
print(f"dataset_normalizer: {timedelta(seconds=end-start)}")

# Create a tokens-only dataset for building word embeddings.
start = timer()
token_extractor(
    dataset_path=DATA_DIR,
    dataset_filename='dataset_norm.csv',
    logging_filename='dataset_norm.csv.log'
    )
end = timer()
print(f"token_extractor: {timedelta(seconds=end-start)}")
