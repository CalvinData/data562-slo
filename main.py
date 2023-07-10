"""
This module implements the core operations of the SLO application
using a setup based on the original SLO tools and MadeWithML's template.
"""
import subprocess
from timeit import default_timer as timer
from datetime import timedelta
from pathlib import Path

# Directories
BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = Path(BASE_DIR, "data")
SRC_DIR = Path(BASE_DIR, "src")

# Preprocess raw tweet dataset.
# Notes - to make this work, we needed to:
# - Convert the raw JSON data UTF-8 (using iconv).
# - Remove some UTF-8 characters that messed up Polyglot/Cld2
#   (for details, see dataset_preprocessor#remove_bad_chars.).
start = timer()
subprocess.run(
    [
        "python",
        Path(SRC_DIR, "dataset_preprocessor.py"),
        f"--json_data_filepath={Path(DATA_DIR, 'slo_rawtweets_20100101-20180510.json')}",
        f"--dataset_path={Path(DATA_DIR)}"
    ],
    check=True)
end = timer()
print(f"dataset_preprocessor: {timedelta(seconds=end-start)}")

# Normalize pre-processed tweet dataset.
start = timer()
subprocess.run(
    [
        "python",
        Path(SRC_DIR, "dataset_normalizer.py"),
        f"--csv_data_filepath={Path(DATA_DIR, 'dataset.csv')}",
        "--post_process=False"
    ],
    check=True)
end = timer()
print(f"dataset_normalizer: {timedelta(seconds=end-start)}")
