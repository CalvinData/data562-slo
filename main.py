"""
This module implements the core operations of the application.
"""
import subprocess
from pathlib import Path

# Directories
BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = Path(BASE_DIR, "data")
SRC_DIR = Path(BASE_DIR, "src")

# Preprocess raw tweet dataset.
subprocess.run(
    [
        "python",
        Path(SRC_DIR, "slo_rawtweet_preprocessor.py"),
        f"--json_data_filepath={Path(DATA_DIR, 'slo_rawtweets_20100101-20180510.json')}",
        f"--dataset_path={Path(DATA_DIR)}"
    ],
    check=True)
